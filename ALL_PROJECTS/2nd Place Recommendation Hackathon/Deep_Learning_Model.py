# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 12:25:06 2020

@author: kbhandari
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import gc
import os
import pickle
import collections
import boto3
import math
import json
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers import BatchNormalization
from keras.layers import LSTM, CuDNNLSTM
from keras.layers import Input
from keras.layers import Dropout
from keras.layers import Reshape
from keras.layers import Concatenate
from keras.layers import GRU, CuDNNGRU
from keras.layers import TimeDistributed
from keras.models import Model
from keras.layers import SpatialDropout1D
from keras.layers import Conv1D, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.layers import Bidirectional
from keras.layers import Average
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,EarlyStopping,Callback
from sklearn.model_selection import KFold
from keras.models import load_model, model_from_json
from keras.regularizers import l2
#from tensorflow.keras import backend as K
from keras import backend as K
from sklearn.metrics import roc_auc_score
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras import initializers, regularizers, constraints
from keras.engine.topology import Layer
import copy
from sklearn import preprocessing

max_sequence_length=60 
batch_size = 512 #int(0.005*len(learn)) #0.005 #2048
epochs = 7
ensemble = False
num_gpu = len(K.tensorflow_backend._get_available_gpus())
feature_importance = False
RunType = 'Model'

wd = "C:/My Desktop/Hackathon/"
os.chdir(wd)

def replace_recency(data):
            columns = data.columns
            cols_to_replace = {}
            for col in columns:
                if 'continuous_sequence' in col.lower() or 'categorical_sequence' in col.lower():
                    cols_to_replace[col] = ''
                else:
                    cols_to_replace[col] = -1 #0
            data = data.fillna(cols_to_replace)
            return data

# Continuous Sequence
def get_continuous_sequence(data, max_sequence_length):
    combined_arrays = None    
    first_time = True
    for col in data.columns:    
        if 'continuous_sequence' in col.lower():
            print(col)
            array = data[col].str.split(",", n = max_sequence_length, expand = True).replace('',np.nan).fillna(value=np.nan).fillna(0)
            if array.shape[1]>max_sequence_length:
                array = array.iloc[:,0:max_sequence_length]
            elif array.shape[1]<max_sequence_length:
                cols_to_add = max_sequence_length-array.shape[1]
                rows_to_add = array.shape[0]
                df = pd.DataFrame(np.zeros((rows_to_add,cols_to_add)))
                array = pd.concat([array, df], axis=1)
            array = np.array(array.astype(np.float))
            array = array.reshape(array.shape[0],array.shape[1],1)
            if first_time:
                combined_arrays = array
                first_time = False
            else:
                combined_arrays = np.concatenate((combined_arrays, array), -1)
    return combined_arrays

def get_max_length(list_of_list):
    max_value = 0
    for l in list_of_list:
        if len(l) > max_value:
            max_value = len(l)
    return max_value

def get_padding(data, categorical_cols, tokenizer = None, max_padding_length = None, max_sequence_length = 80):
    # Tokenize Sentences
    word_index, max_length, padded_docs, word_tokenizer = {},{},{},{}
    if tokenizer is None:    
        for col in categorical_cols:            
            print("Processing column:", col)        
            t = Tokenizer()
            t.fit_on_texts(data[col].astype(str))
            word_index[col] = t.word_index
            txt_to_seq = t.texts_to_sequences(data[col].astype(str))
            max_length_value = get_max_length(txt_to_seq)
            max_length[col] = max_length_value if max_length_value < max_sequence_length else max_sequence_length #50
            padded_docs[col] = pad_sequences(txt_to_seq, maxlen=max_length[col], padding='post')
            word_tokenizer[col] = t
        return word_index, max_length, padded_docs, word_tokenizer

    else:
        for col in categorical_cols:
            print("Processing column:", col)
            t = tokenizer[col]
            txt_to_seq = t.texts_to_sequences(data[col].astype(str))
            padded_docs[col] = pad_sequences(txt_to_seq, maxlen=max_padding_length[col], padding='post')        
        return padded_docs       

def dot_product(x, kernel):
    """
    Wrapper for dot product operation, in order to be compatible with both
    Theano and Tensorflow
    Args:
        x (): input
        kernel (): weights
    Returns:
    """
    if K.backend() == 'tensorflow':
        # todo: check that this is correct
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)

class Attention(Layer):
    def __init__(self,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True,
                 return_attention=False,
                 **kwargs):
        """
        Keras Layer that implements an Attention mechanism for temporal data.
        Supports Masking.
        Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Note: The layer has been tested with Keras 1.x
        Example:

            # 1
            model.add(LSTM(64, return_sequences=True))
            model.add(Attention())
            # next add a Dense layer (for classification/regression) or whatever...
            # 2 - Get the attention scores
            hidden = LSTM(64, return_sequences=True)(words)
            sentence, word_scores = Attention(return_attention=True)(hidden)
        """
        self.supports_masking = True
        self.return_attention = return_attention
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 #shape=(input_shape[-1], input_shape[1]),
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight(shape=(input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     #shape=(input_shape[-1],),
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        eij = dot_product(x, self.W)

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        weighted_input = x * K.expand_dims(a)

        result = K.sum(weighted_input, axis=1)

        if self.return_attention:
            return [result, a]
        return result

    def compute_output_shape(self, input_shape):
        if self.return_attention:
            return [(input_shape[0], input_shape[-1]),
                    (input_shape[0], input_shape[1])]
        else:
            return input_shape[0], input_shape[-1]

# Loss Function  
gamma = 2.0
epsilon = K.epsilon()
def focal_loss(y_true, y_pred):
    pt = y_pred * y_true + (1-y_pred) * (1-y_true)
    pt = K.clip(pt, epsilon, 1-epsilon)
    CE = -K.log(pt)
    FL = K.pow(1-pt, gamma) * CE
    loss = K.sum(FL, axis=1)
    return loss

# Model
def create_model(categorical_cols, numerical_cols, word_index, max_length, final_layer, continuous_sequence, max_sequence_length, max_sequence_features, num_gpu=1, cpu_model=False, ensemble=False):

    inputs = []
    embeddings_1, embeddings_2, embeddings_3 = [], [], []

    # Embedding for categorical columns:
    for col in categorical_cols:

        input_cat_cols = Input(shape=(max_length[col],))
        inputs.append(input_cat_cols)

        vocab_size = len(word_index[col]) + 1
        embed_size = int(np.min([np.ceil((vocab_size)/2), 50])) #25
        embedding = Embedding(vocab_size, embed_size, input_length=max_length[col], name='{0}_embedding_1'.format(col), trainable=True)(input_cat_cols)
        embedding = SpatialDropout1D(0.2)(embedding) #0.25
        if max_length[col] > 30:
            if num_gpu > 0:
                embedding = Bidirectional(CuDNNGRU(max_length[col], return_sequences=True))(embedding)
                embedding = Bidirectional(CuDNNGRU(max_length[col], return_sequences=True))(embedding)
            else:
                if cpu_model:
                    embedding = Bidirectional(GRU(max_length[col], return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(embedding)
                else:
                    embedding = Bidirectional(GRU(max_length[col], return_sequences=True, reset_after=True, recurrent_activation='sigmoid', implementation=1))(embedding)
                    embedding = Bidirectional(GRU(max_length[col], return_sequences=True, reset_after=True, recurrent_activation='sigmoid', implementation=1))(embedding)
            embedding=Attention()(embedding)
        else:
            embedding = Reshape(target_shape=(embed_size*max_length[col],))(embedding)
        embeddings_1.append(embedding)

        if ensemble:
            vocab_size = len(word_index[col]) + 1
            embed_size = int(np.min([np.ceil((vocab_size)/2), 50])) #25
            embedding = Embedding(vocab_size, embed_size, input_length=max_length[col], name='{0}_embedding_2'.format(col), trainable=True)(input_cat_cols)
            embedding = SpatialDropout1D(0.2)(embedding) #0.25
            if max_length[col] > 30:
                if num_gpu > 0:
                    embedding = Bidirectional(CuDNNGRU(max_length[col], return_sequences=True))(embedding)
                else:
                    if cpu_model:
                        embedding = Bidirectional(GRU(max_length[col], return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(embedding)
                    else:
                        embedding = Bidirectional(GRU(max_length[col], return_sequences=True, reset_after=True, recurrent_activation='sigmoid', implementation=1))(embedding)
                embedding=Attention()(embedding)
            else:
                embedding = Reshape(target_shape=(embed_size*max_length[col],))(embedding)
            embeddings_2.append(embedding)      

            vocab_size = len(word_index[col]) + 1
            embed_size = int(np.min([np.ceil((vocab_size)/2), 50])) #25
            embedding = Embedding(vocab_size, embed_size, input_length=max_length[col], name='{0}_embedding_3'.format(col), trainable=True)(input_cat_cols)
            embedding = SpatialDropout1D(0.2)(embedding) #0.25
            if max_length[col] > 30:
                if num_gpu > 0:
                    embedding = Bidirectional(CuDNNGRU(embed_size, return_sequences=True))(embedding)
                else:
                    if cpu_model:
                        embedding = Bidirectional(GRU(embed_size, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(embedding)
                    else:
                        embedding = Bidirectional(GRU(embed_size, return_sequences=True, reset_after=True, recurrent_activation='sigmoid', implementation=1))(embedding)
                embedding=Attention()(embedding)
            else:
                embedding = Reshape(target_shape=(embed_size*max_length[col],))(embedding)
            embeddings_3.append(embedding)

    if continuous_sequence:
        input_array = Input(shape=(max_sequence_length, max_sequence_features))
        inputs.append(input_array)
        if num_gpu > 0:
            RNN = CuDNNLSTM(max_length[col], return_sequences=True)(input_array)
            RNN = CuDNNLSTM(max_length[col], return_sequences=True)(RNN)
        else:
            if cpu_model:
                RNN = LSTM(max_length[col], dropout=0.1, recurrent_dropout=0.1, return_sequences=True)(input_array)
            else:
                RNN = LSTM(max_length[col], return_sequences=True)(input_array)
                RNN = LSTM(max_length[col], return_sequences=True)(RNN)
        RNN=Attention()(RNN)
        embeddings_1.append(RNN)

        if ensemble:
            if num_gpu > 0:
                RNN = CuDNNLSTM(max_length[col], return_sequences=True)(input_array)
            else:
                if cpu_model:
                    RNN = LSTM(max_length[col], dropout=0.2, recurrent_dropout=0.2, return_sequences=True)(input_array)
                else:
                    RNN = LSTM(max_length[col], return_sequences=True)(input_array)
            RNN=Attention()(RNN)
            embeddings_2.append(RNN)

            if num_gpu > 0:
                RNN = CuDNNLSTM(max_length[col], return_sequences=True)(input_array)
            else:
                if cpu_model:
                    RNN = LSTM(max_length[col], dropout=0.2, recurrent_dropout=0.2, return_sequences=True)(input_array)
                else:
                    RNN = LSTM(max_length[col], return_sequences=True)(input_array)
            RNN=Attention()(RNN)
            embeddings_3.append(RNN)

    if len(numerical_cols)>0:    
        # Dense layer for continous variables 1
        input_num_cols = Input(shape=(len(numerical_cols),))
        inputs.append(input_num_cols)

        numeric = BatchNormalization()(input_num_cols)
        numeric = Dense(100, activation='relu', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(numeric) #50
        numeric = Dropout(.3)(numeric)
        numeric = BatchNormalization()(numeric)
        numeric = Dense(50, activation='relu', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(numeric) #30
        embeddings_1.append(numeric)

        if ensemble:
            # Dense layer for continous variables 2
#             bn0 = BatchNormalization()(input_num_cols)
            numeric = Dense(100, activation='relu', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(input_num_cols) #50
            numeric = Dropout(.3)(numeric)
            numeric = Dense(59, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(numeric) #30
            numeric = Dropout(.2)(numeric)
            embeddings_2.append(numeric)

            # Dense layer for continous variables 3
#             bn0 = BatchNormalization()(input_num_cols)
            numeric = Dense(100, activation='relu', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(input_num_cols) #50
            numeric = Dropout(.3)(numeric)
            numeric = Dense(50, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(numeric) #30
            numeric = Dropout(.2)(numeric)
            embeddings_3.append(numeric)

    x1 = Concatenate()(embeddings_1)
    if ensemble:
        x2 = Concatenate()(embeddings_2)
        x3 = Concatenate()(embeddings_3)

    x1 = BatchNormalization()(x1)
    x1 = Dense(30, activation='relu')(x1)
    x1 = Dropout(.2)(x1)
    x1 = BatchNormalization()(x1)
    x1 = Dense(15, activation='relu')(x1)
    output1 = Dense(final_layer, activation='sigmoid', name='model_1')(x1)   

    if ensemble:
    #     x2 = BatchNormalization()(x2)
        x2 = Dense(500, activation='relu')(x2)
        x2 = Dropout(.4)(x2)
    #     x2 = BatchNormalization()(x2)
        x2 = Dense(300, activation='relu')(x2)
        x2 = Dropout(.3)(x2)
    #     x2 = BatchNormalization()(x2)
        x2 = Dense(50, activation='relu')(x2)
        output2 = Dense(final_layer, activation='sigmoid', name='model_2')(x2) 

    #     x3 = BatchNormalization()(x3)
        x3 = Dense(500, activation='relu')(x3)
        x3 = Dropout(.4)(x3)
    #     bn3 = BatchNormalization()(x3)
        x3 = Dense(200, activation='relu')(x3)
        x3 = Dropout(.3)(x3)
    #     x3 = BatchNormalization()(x3)
        x3 = Dense(50, activation='relu')(x3)
        output3 = Dense(final_layer, activation='sigmoid', name='model_3')(x3) 

    if ensemble:
        model = Model(inputs, [output1, output2, output3], name='sequential_model')
    else:
        model = Model(inputs, [output1], name='sequential_model')
#     model.compile(loss = [focal_loss,focal_loss,focal_loss], optimizer = "adam", metrics=['accuracy'])
    return model

if RunType.upper() == 'MODEL':
    data = pd.read_csv(wd + "DL_MDL_UNIV.csv", converters={'customer_id':str})
    data = replace_recency(data)
    missing = data.isna().sum()
    
    # Train-Test Split
    learn, validation = train_test_split(data, test_size = 0.05, random_state = 0)
#    del data
#    gc.collect()
    
    print("Learn Shape:", learn.shape)
    print("Validation Shape:", validation.shape)
    
    # Variable Configuration
    sub = pd.read_csv('submission.csv')
    dependent_variables = ["Target"]  #[x for x in sub.columns[1:]]
    #dependent_variables = [col for col in learn.columns if col not in ['customer_id'] and '_sequence_' not in col]
    Total_Dependent_Variables = len(dependent_variables)
    cat_cols = [col for col in learn.columns if learn[col].dtypes == 'O' and 'continuous_sequence_' not in col] #and 'customer_id' not in col]
    cols_to_exclude = dependent_variables + cat_cols# + ['customer_id']
    continuous_cols = [col for col in learn.columns if col not in cols_to_exclude and '_sequence_' not in col]
    
    word_index, max_length, padded_docs, tokenizer = get_padding(data = learn, categorical_cols = cat_cols, max_sequence_length = max_sequence_length, max_padding_length = None, tokenizer = None)
    print("Max Length of Categorical Variables: ", max_length)
           
    continuous_sequence_learn = get_continuous_sequence(learn, max_sequence_length=max_sequence_length)
    if continuous_sequence_learn is not None:
        print(continuous_sequence_learn.shape[-1])
        max_sequence_features = continuous_sequence_learn.shape[-1]
    else:
        max_sequence_features = 0
    
    pickle_byte_obj = [word_index, max_length, tokenizer, cat_cols, continuous_cols, max_sequence_length, max_sequence_features]
    pickle.dump(pickle_byte_obj, open(wd+"data_config.pkl", "wb"))
    
    padded_docs_validation = get_padding(data = validation, categorical_cols = cat_cols, tokenizer = tokenizer, max_padding_length = max_length, max_sequence_length = None)
    continuous_sequence_validation = get_continuous_sequence(validation, max_sequence_length=max_sequence_length)
    
    print("Padding Done")
    
    
    input_list_learn, input_list_validation = [], []
    for col in cat_cols:
        input_list_learn.append(padded_docs[col])
        input_list_validation.append(padded_docs_validation[col])
    if continuous_sequence_learn is not None:
        continuous_sequence = True
        input_list_learn.append(continuous_sequence_learn)
        input_list_validation.append(continuous_sequence_validation)
    else:
        continuous_sequence = False
    if len(continuous_cols)>0:
        min_max_scaler = preprocessing.MinMaxScaler()
        learn_continuous = min_max_scaler.fit_transform(learn[continuous_cols])
        validation_continuous = min_max_scaler.transform(validation[continuous_cols])
        input_list_learn.append(learn_continuous)
        input_list_validation.append(validation_continuous)
    
    del padded_docs, padded_docs_validation
    if continuous_sequence_learn is not None:
        del continuous_sequence_learn, continuous_sequence_validation
    gc.collect()
    
    y_train = learn[dependent_variables].values
    y_val = validation[dependent_variables].values
    
    
    def setup_multi_gpu(model):
    
        import tensorflow as tf
        from keras.utils.training_utils import multi_gpu_model
        from tensorflow.python.client import device_lib
    
        # IMPORTANT: Tells tf to not occupy a specific amount of memory
        from keras.backend.tensorflow_backend import set_session
    
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
        sess = tf.Session(config=config)
        set_session(sess)  # set this TensorFlow session as the default session for Keras.
    
        print('reading gpus avaliable..')
        local_device_protos = device_lib.list_local_devices()
        avail_gpus = [x.name for x in local_device_protos if x.device_type == 'GPU']
        num_gpu = len(avail_gpus)
        print('Amount of GPUs available: %s' % num_gpu)
    
        multi_model = multi_gpu_model(model, gpus=num_gpu)
    
        return multi_model 
    
    if num_gpu>1:
        _model = create_model(cat_cols, continuous_cols, word_index, max_length, Total_Dependent_Variables, continuous_sequence, max_sequence_length, max_sequence_features, num_gpu, cpu_model=False, ensemble=ensemble)
        print(_model.summary())
        model = setup_multi_gpu(_model)
    else:
        model = create_model(cat_cols, continuous_cols, word_index, max_length, Total_Dependent_Variables, continuous_sequence, max_sequence_length, max_sequence_features, num_gpu, cpu_model=True, ensemble=ensemble)
        print(model.summary())
    
    if ensemble:
    #         model.compile(loss = [focal_loss,focal_loss,focal_loss], optimizer = "adam", metrics=['accuracy'])
        model.compile(loss = ['binary_crossentropy','binary_crossentropy','binary_crossentropy'], optimizer = "adam", metrics=['accuracy'])
    else:
    #         model.compile(loss = [focal_loss], optimizer = "adam", metrics=['accuracy'])
        model.compile(loss = ['binary_crossentropy'], optimizer = "adam", metrics=['accuracy'])
    reduce_learning_rate = ReduceLROnPlateau(monitor = "val_accuracy", factor = 0.6, patience=1, min_lr=1e-4,  verbose=1, mode = 'max')
    early_stopping = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=2, restore_best_weights = True) #min_delta=0.0000001
    filepath="model_checkpoint.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [reduce_learning_rate, early_stopping]
    
    # Class Weights
    def create_class_weight(labels_dict, mu=0.15, default=False):
        total = np.sum(list(labels_dict.values()))
        keys = labels_dict.keys()
        class_weight = dict()
    
        if default == False:
            for key in keys:
                if labels_dict[key] == 0:
                    labels_dict[key] = 1
                score = math.log(mu*total/float(labels_dict[key]))
                class_weight[key] = score if score > 1.0 else 1.0
        else:
            for key in keys:
                class_weight[key] = 1.0
    
        return class_weight
    
    #labels_dict = {0: 2813, 1: 78, 2: 2814, 3: 78, 4: 7914, 5: 248, 6: 7914, 7: 248}
    labels_dict = dict(pd.Series(np.sum(y_train, axis=0)))
    class_weights = create_class_weight(labels_dict, mu=0.3, default=True) # Change to False to use class weights
    print("Class Weights: ", class_weights)
    
    print("Building Model")
    if ensemble:
        entity_embedding_model = model.fit(input_list_learn,[y_train,y_train,y_train], 
                                 validation_data=(input_list_validation,[y_val,y_val,y_val]), 
                                 epochs=epochs, #25
                                 callbacks=callbacks_list, 
                                 shuffle=True,#False 
                                 batch_size=batch_size, #2056
                                 verbose=1,
                                 class_weight = class_weights)
    else:
        entity_embedding_model = model.fit(input_list_learn,[y_train], 
                                 validation_data=(input_list_validation,[y_val]), 
                                 epochs=epochs, #25
                                 callbacks=callbacks_list, 
                                 shuffle=True,#False 
                                 batch_size=batch_size, #2056
                                 verbose=1)
                
    
    if num_gpu>1:
        _model.save(wd + 'model.h5')
        _model.save_weights(wd + 'model_weights.h5')
        
    else:
        model.save(wd + 'model.h5')
        model.save_weights(wd + 'model_weights.h5')
    
    new_model = load_model(wd + 'model.h5', custom_objects={'Attention': Attention, 'focal_loss': focal_loss})
    
    #architecture = entity_embedding_model.model.to_json()
    #weights = entity_embedding_model.model.get_weights()
    #
    #pickle.dump(architecture, open(wd+"keras-model.json", "wb"))
    #pickle.dump(weights, open(wd+"keras-weights.pkl", "wb"))
    #
    #loaded_model = pickle.load(open(wd+"keras-model.json", "rb"))
    #loaded_weights = pickle.load(open(wd+"keras-weights.pkl", "rb"))
    #
    #use_rnn = np.max(list(max_length.values()))
    #if use_rnn > 30 or continuous_sequence_learn is not None:    
    #    new_model = model_from_json(loaded_model, custom_objects={'Attention': Attention, 'focal_loss': focal_loss})
    #    new_model.set_weights(loaded_weights)
    #else:
    #    new_model = model_from_json(loaded_model)
    #    new_model.set_weights(loaded_weights)      
    
    print("Scoring")
    predictions = new_model.predict(input_list_learn,verbose=1, batch_size=4096)
    if ensemble:
        learn_preds = np.sum(predictions, axis = 0)/len(predictions)
        learn_preds = pd.DataFrame(learn_preds, columns = dependent_variables)
    else:
        learn_preds = pd.DataFrame(predictions, columns = dependent_variables)
#    learn_preds['customer_id'] = learn['customer_id'].reset_index(drop=True)
    learn_preds[['customer_id', 'item_id']] = learn[['customer_id', 'item_id']].reset_index(drop=True)
    
    predictions = new_model.predict(input_list_validation,verbose=1, batch_size=4096)
    if ensemble:
        validation_preds = np.sum(predictions, axis = 0)/len(predictions)
        validation_preds = pd.DataFrame(validation_preds, columns = dependent_variables)
    else:
        validation_preds = pd.DataFrame(predictions, columns = dependent_variables)
#    validation_preds['customer_id'] = validation['customer_id'].reset_index(drop=True)
        validation_preds[['customer_id', 'item_id']] = validation[['customer_id', 'item_id']].reset_index(drop=True)


# Test Data
test = pd.read_csv(wd + "DL_MDL_UNIV_SCORE.csv", converters={'customer_id':str})
sub = pd.read_csv('submission.csv')
dependent_variables=['Target']#[x for x in sub.columns[1:]]

test = replace_recency(test)

data_config = pickle.load(open(wd+"data_config.pkl", "rb"))
word_index, max_length, tokenizer, cat_cols, continuous_cols, max_sequence_length, max_sequence_features = data_config[0], data_config[1], data_config[2], data_config[3], data_config[4], data_config[5], data_config[6]
padded_docs_test = get_padding(test, cat_cols, tokenizer, max_length)
continuous_sequence_test = get_continuous_sequence(test, max_sequence_length=max_sequence_length)
print("Padding Done")

input_list_test = []
for col in cat_cols:
    input_list_test.append(padded_docs_test[col])
if continuous_sequence_test is not None:
    continuous_sequence = True
    input_list_test.append(continuous_sequence_test)
else:
    continuous_sequence = False
if len(continuous_cols)>0:
    test_continuous = min_max_scaler.transform(test[continuous_cols])
    input_list_test.append(test_continuous)

del padded_docs_test
if continuous_sequence_test is not None:
    del continuous_sequence_test
gc.collect()

#loaded_model = pickle.load(open(wd+"keras-model.json", "rb"))
#loaded_weights = pickle.load(open(wd+"keras-weights.pkl", "rb"))

print("Scoring")    
#    if num_gpu>0:
new_model = load_model('model.h5', custom_objects={'Attention': Attention, 'focal_loss': focal_loss})
#    else:        
#        new_model = create_model(cat_cols, continuous_cols, word_index, max_length, Total_Dependent_Variables, continuous_sequence, max_sequence_length, max_sequence_features, num_gpu, cpu_model=False, ensemble=False)
#        new_model.load_weights('model_weights.h5')
    
predictions = new_model.predict(input_list_test,verbose=1, batch_size=4096)
test_preds = pd.DataFrame(predictions, columns = dependent_variables)
#test_preds['customer_id'] = test['customer_id'].reset_index(drop=True)
test_preds[['customer_id', 'item_id']] = test[['customer_id', 'item_id']].reset_index(drop=True)
test_preds = pd.pivot_table(test_preds, index='customer_id', columns=['item_id'], aggfunc=max)
test_preds.columns=test_preds.columns.map('_'.join).str.strip('_')    
test_preds.reset_index(level=['customer_id'],inplace=True)
test_preds['customer_id'] = test_preds['customer_id'].astype(float).astype(int)

#popularity = pd.DataFrame(data[dependent_variables].mean()).reset_index(drop=False)
popularity = dict(data[dependent_variables].mean())
popularity['Target_85123A'] = 1
popularity['Target_85099B'] = 0.9
popularity['Target_84879'] = 0.8

sub = pd.read_csv(wd + 'FINAL/' + 'GOD_IS_GREAT.csv')
cid = sub[~sub['customer_id'].isin(test_preds['customer_id'])]
cid = cid[test_preds.columns]

for key, value in popularity.items():
    cid[key] = value

mods = pd.concat([test_preds, cid], axis=0)
mods = mods[mods['customer_id'].isin(sub['customer_id'])]
mods = mods[sub.columns]
mods = mods.sort_values(by="customer_id")
mods.to_csv(wd + "predictions_KB_DL6.csv", index=False)










if feature_importance:

    # Feature Importance
    test = pd.read_csv(wd + "DL_MDL_UNIV.csv")
    sub = pd.read_csv('submission.csv')
    dependent_variables=[x for x in sub.columns[1:]]
    
    test = replace_recency(test)
    
    data_config = pickle.load(open(wd+"data_config.pkl", "rb"))
    word_index, max_length, tokenizer, cat_cols, continuous_cols, max_sequence_length, max_sequence_features = data_config[0], data_config[1], data_config[2], data_config[3], data_config[4], data_config[5], data_config[6]
    padded_docs_test = get_padding(test, cat_cols, tokenizer, max_length)
    continuous_sequence_test = get_continuous_sequence(test, max_sequence_length=max_sequence_length)
    print("Padding Done")
    
    input_list_test = []
    for col in cat_cols:
        input_list_test.append(padded_docs_test[col])
    if continuous_sequence_test is not None:
        continuous_sequence = True
        input_list_test.append(continuous_sequence_test)
    else:
        continuous_sequence = False
    if len(continuous_cols)>0:
        test_continuous = min_max_scaler.transform(test[continuous_cols])
        input_list_test.append(test_continuous)
    
    del padded_docs_test
    if continuous_sequence_test is not None:
        del continuous_sequence_test
    gc.collect()
    
    print("Scoring")    
    new_model = load_model('model.h5', custom_objects={'Attention': Attention, 'focal_loss': focal_loss})
        
    y_test = test[dependent_variables].values
    predictions = new_model.predict(input_list_test,verbose=1, batch_size=1024)
    loss, accuracy = new_model.evaluate(input_list_test, y_test, batch_size=1024)
    
    table_of_error = pd.DataFrame({'Feature': ['Baseline'], 'Loss': [loss], 'accuracy': [accuracy]})
    
    "Feature Importance Calculation"
    for n, col in enumerate(cat_cols):
        perturbed_data = copy.deepcopy(input_list_test)
        np.random.shuffle(perturbed_data[n])
        loss, accuracy = new_model.evaluate(perturbed_data, y_test, batch_size=4096)
        table_of_error = table_of_error.append({'Feature': col, 'Loss': loss, 'accuracy': accuracy}, ignore_index=True)
    
    array_index = n+1
    
    if continuous_sequence:
        continuous_sequence_cols = [col for col in test.columns if 'continuous_sequence' in col.lower()]
        for n, col in enumerate(continuous_sequence_cols):
            perturbed_data = copy.deepcopy(input_list_test)
            tmp_slice = copy.deepcopy(perturbed_data[array_index][:, :, n])
            np.random.shuffle(tmp_slice)
            perturbed_data[array_index][:,:,n] = tmp_slice
            loss, accuracy = new_model.evaluate(perturbed_data, y_test, batch_size=4096)
            table_of_error = table_of_error.append({'Feature': col, 'Loss': loss, 'accuracy': accuracy}, ignore_index=True)
    
    array_index += 1
    
    if len(continuous_cols)>0:
        for n, col in enumerate(continuous_cols):
            print(col)
            perturbed_data = copy.deepcopy(input_list_test)
            perturbed_data[array_index] = pd.DataFrame(perturbed_data[array_index], columns = continuous_cols)
            perturbed_data[array_index][col] = perturbed_data[array_index][col].sample(frac=1).values
            loss, accuracy = new_model.evaluate(perturbed_data, y_test, batch_size=4096)
            table_of_error = table_of_error.append({'Feature': col, 'Loss': loss, 'accuracy': accuracy}, ignore_index=True)
            
    Baseline_loss = table_of_error.loc[table_of_error.Feature=="Baseline"]['Loss'].values        
    table_of_error['Error'] = Baseline_loss - table_of_error['Loss']
    table_of_error = table_of_error.loc[table_of_error.Feature != "Baseline"]
    table_of_error = table_of_error.sort_values(by="Error", ascending=True)
    pickle.dump(table_of_error, open(wd+"table_or_error.pkl", "wb"))        
