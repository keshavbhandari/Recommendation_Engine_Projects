# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 07:34:50 2019

@author: kbhandari
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import gc
import re
import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Layer
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers import BatchNormalization
from keras.layers import LSTM
from keras.layers import Input
from keras.layers import Dropout
from keras.layers import Reshape
from keras.layers import Concatenate
from keras.layers import GRU
from keras.models import Model
from keras.layers import SpatialDropout1D
from keras.layers import Conv1D, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.layers import Bidirectional
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,EarlyStopping
from keras import backend as K
from sklearn.model_selection import KFold
from keras.models import Sequential, load_model
#from tensorflow.keras import backend as K

wd = "C:/Users/kbhandari/OneDrive - Epsilon/Desktop/Not Just OLS/Food Delivery Time/Participants Data/"
os.chdir(wd)
data = pd.read_excel(wd + "Data_Train.xlsx", na_values = '-', encoding="utf-8")
test = pd.read_excel(wd + "Data_Test.xlsx", na_values = '-', encoding="utf-8")

#from itertools import chain
#
## return list from series of comma-separated strings
#def chainer(s):
#    return list(chain.from_iterable(s.str.split(',')))
#
## calculate lengths of splits
#lens = data['Cuisines'].str.split(',').map(len)
#
## create new dataframe, repeating or chaining as appropriate
#res = pd.DataFrame({'Restaurant': np.repeat(data['Restaurant'], lens),
#                    'Location': np.repeat(data['Location'], lens),
#                    'Cuisines': chainer(data['Cuisines']),
#                    'Average_Cost': np.repeat(data['Average_Cost'], lens),
#                    'Minimum_Order': np.repeat(data['Minimum_Order'], lens),
#                    'Rating': np.repeat(data['Rating'], lens),
#                    'Votes': np.repeat(data['Votes'], lens),
#                    'Reviews': np.repeat(data['Reviews'], lens),
#                    'Delivery_Time': np.repeat(data['Delivery_Time'], lens),
#                    })
#
#res['Cuisines'] = res['Cuisines'].replace(" ", "", regex = True)
#data = data.append(res,sort=False).reset_index(drop=True)

#def data_augmentation(data, column):
#    lens = pd.DataFrame(data[column].str.split(',', expand = True))
#    lens = lens.values
#    aug = pd.DataFrame(np.apply_along_axis(np.random.permutation, 1, lens))
#    aug = aug.fillna('')
#    aug[column] = aug[aug.columns].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
#    aug[column] = aug[column].str.strip()
#    aug[column] = aug[column].replace('\s+', ' ', regex=True)
#    aug = aug[column]
#    data_copy = data.copy()
#    data_copy[column] = aug
##    data = data.append(data_copy,sort=False).reset_index(drop=True)
#    return data_copy
#
#for i in range(1):
#    data_aug_1 = data_augmentation(data, 'Cuisines')
#    data_aug_2 = data_augmentation(data, 'Location')
#    data = data.append([data_aug_1, data_aug_2],sort=False).reset_index(drop=True)
#    del data_aug_1, data_aug_2
#    gc.collect()
    
#data.drop_duplicates(subset=None, keep='first', inplace=True)


def data_preprocess(data):
    data[['Average_Cost', 'Minimum_Order', 'Rating']] = data[['Average_Cost', 'Minimum_Order', 'Rating']].replace('[^\w\s]','',regex=True)
    data['New'] = np.where(data['Rating']=='NEW', 1, 0)
    data['Opening_Soon'] = np.where(data['Rating']=='Opening Soon', 1, 0)
    data['Temporarily Closed'] = np.where(data['Rating']=='Temporarily Closed', 1, 0)
    data = data.replace('NEW', None)
    data = data.replace('Opening Soon', None)
    data = data.replace('Temporarily Closed', None)
    data['Average_Cost'] = np.where(data['Average_Cost']=='for', data['Average_Cost'].mode()[0], data['Average_Cost'])
    
    data[['Average_Cost', 'Minimum_Order', 'Rating']] = data[['Average_Cost', 'Minimum_Order', 'Rating']].apply(pd.to_numeric) 
    data = data.replace(np.nan, -1)
    data['Location'] = data['Location'].astype(str).str.lower()
    data = data.replace(',',' ', regex=True)
    data = data.replace('-',' ', regex=True)
    data = data.replace('[.]',' ', regex=True)
    data = data.replace('gurgoan', 'gurgaon', regex=True)
    data = data.replace('  ',' ', regex=True)
    data['Count_of_Cuisines'] = data['Cuisines'].str.split().str.len()
    return data

data = data_preprocess(data)
nrow_data = len(data)
test = data_preprocess(test)
Delivery_Time = pd.get_dummies(pd.DataFrame(data['Delivery_Time']), prefix='', prefix_sep = '')


train_test_combined = data.append(test,sort=False).reset_index(drop=True)


#cat_cols = ['Restaurant', 'Location', 'Cuisines', 'Average_Cost', 'Minimum_Order', 'Count_of_Cuisines']
cat_cols = ['Location', 'Cuisines', 'Average_Cost', 'Minimum_Order', 'Count_of_Cuisines']
def get_padding(data, categorical_cols):
    # Tokenize Sentences
    word_index, max_length, padded_docs = {},{},{}
    for col in cat_cols:
        print("Processing column:", col)
        t = Tokenizer()
        t.fit_on_texts(data[col].astype(str))
        word_index[col] = t.word_index
        txt_to_seq = t.texts_to_sequences(data[col].astype(str))
        max_length[col] = len(max(txt_to_seq, key = lambda x: len(x)))
        padded_docs[col] = pad_sequences(txt_to_seq, maxlen=max_length[col], padding='post')
    
    return word_index, max_length, padded_docs
    
word_index, max_length, padded_docs = get_padding(train_test_combined, cat_cols)
del train_test_combined
gc.collect()

continuous_cols = ['Rating', 'Votes', 'Reviews', 'New', 'Opening_Soon', 'Temporarily Closed', 'Count_of_Cuisines']
train_continuous = data[continuous_cols]
test_continuous = test[continuous_cols]

class attention(Layer):
    def __init__(self,**kwargs):
        super(attention,self).__init__(**kwargs)

    def build(self,input_shape):
        self.W=self.add_weight(name="att_weight",shape=(input_shape[-1],1),initializer="normal")
        self.b=self.add_weight(name="att_bias",shape=(input_shape[1],1),initializer="zeros")        
        super(attention, self).build(input_shape)

    def call(self,x):
        et=K.squeeze(K.tanh(K.dot(x,self.W)+self.b),axis=-1)
        at=K.softmax(et)
        at=K.expand_dims(at,axis=-1)
        output=x*at
        return K.sum(output,axis=1)

    def compute_output_shape(self,input_shape):
        return (input_shape[0],input_shape[-1])

    def get_config(self):
        return super(attention,self).get_config()

import tensorflow as tf
from keras import backend as K, initializers, regularizers, constraints
from keras.engine.topology import Layer


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

def create_model(categorical_cols, numerical_cols, word_index, max_length):
    
    inputs = []
    embeddings = []
    
    # Embedding for categorical columns:
    for col in categorical_cols:
        if col in ['Location', 'Cuisines']:
             # LSTM
            vocab_size = len(word_index[col]) + 1
            input_cat_cols = Input(shape=(max_length[col],))
            embed_size = int(min(np.ceil((vocab_size)/2), 15)) #25
            embedding = Embedding(vocab_size, embed_size, input_length=max_length[col], name='{0}_embedding'.format(col), trainable=True)(input_cat_cols)
            embedding = SpatialDropout1D(0.15)(embedding) #0.25
            lstm = LSTM(embed_size, return_sequences=True)(embedding) #dropout=0.2, recurrent_dropout=0.2
            lstm=Attention()(lstm)
            inputs.append(input_cat_cols)
            embeddings.append(lstm)
            
             # Convoluted LSTM (Best)
#            vocab_size = len(word_index[col]) + 1
#            input_cat_cols = Input(shape=(max_length[col],))
#            embed_size = int(min(np.ceil((vocab_size)/2), 15)) #25
#            embedding = Embedding(vocab_size, embed_size, input_length=max_length[col], name='{0}_embedding'.format(col), trainable=True)(input_cat_cols)
#            embedding = SpatialDropout1D(0.15)(embedding) #0.25
#            cnn = Conv1D(filters = embed_size, kernel_size=3, activation='relu')(embedding)
#            cnn = GlobalMaxPooling1D()(cnn)
#            inputs.append(input_cat_cols)
#            embeddings.append(cnn)
            
            #Bidirectional LSTM
#            vocab_size = len(word_index[col]) + 1
#            input_cat_cols = Input(shape=(max_length[col],))
#            embed_size = int(min(np.ceil((vocab_size)/2), 15)) #25
#            embedding = Embedding(vocab_size, embed_size, input_length=max_length[col], name='{0}_embedding'.format(col), trainable=True)(input_cat_cols)
#            embedding = SpatialDropout1D(0.15)(embedding) #0.25
#            gru1 = Bidirectional(GRU(embed_size, return_sequences=True))(embedding)
#            gru2 = Bidirectional(GRU(7))(gru1)
#            inputs.append(input_cat_cols)
#            embeddings.append(gru2)

        else:
            vocab_size = len(word_index[col]) + 1
            input_cat_cols = Input(shape=(max_length[col],))
            embed_size = int(min(np.ceil((vocab_size)/2), 15)) #25
            embedding = Embedding(vocab_size, embed_size, input_length=max_length[col], name='{0}_embedding'.format(col), trainable=True)(input_cat_cols)
            embedding = SpatialDropout1D(0.15)(embedding) #0.25
            embedding = Reshape(target_shape=(embed_size*max_length[col],))(embedding)
            inputs.append(input_cat_cols)
            embeddings.append(embedding)
    
    # Dense layer for continous variables for item description column
    input_num_cols = Input(shape=(len(numerical_cols),))
    bn0 = BatchNormalization()(input_num_cols)
    numeric = Dense(30, activation='relu')(bn0) #50
    numeric = Dropout(.20)(numeric)
    numeric = Dense(15)(numeric) #30
    inputs.append(input_num_cols)
    embeddings.append(numeric)
    
    x = Concatenate()(embeddings)
    
    bn1 = BatchNormalization()(x)
    x = Dense(100, activation='relu')(bn1)
    x = Dropout(.2)(x)
    bn2 = BatchNormalization()(x)
    x = Dense(50, activation='relu')(bn2)
    x = Dropout(.2)(x)
    bn3 = BatchNormalization()(x)
    x = Dense(25, activation='relu')(bn3)
    x = Dropout(.2)(x)
    bn4 = BatchNormalization()(x)
    x = Dense(15, activation='relu')(bn4)
    x = BatchNormalization()(x)
    output = Dense(7, activation='sigmoid')(x)   
    model = Model(inputs, output)
    model.compile(loss = 'binary_crossentropy', optimizer = "adam", metrics=['accuracy'])    
#     print(model.summary())
    return model



num_cols = train_continuous.columns.to_list()
splits = 10
file_path = wd + "/model"
directory = os.path.dirname(file_path)
print(file_path)
try:
    os.stat(file_path)
except:
    os.mkdir(file_path) 
    
from glob import glob
glob('/model/*')

test_preds = np.zeros((test.shape[0],7))
kf = KFold(n_splits=splits)
fold_number = 1
predict_test = 0
for train_index, validation_index in kf.split(data.index, data.Delivery_Time):
    print("Fold Number:",fold_number)
    input_list_train, input_list_val, input_list_test = [], [], []
    for col in cat_cols:
        input_list_train.append(padded_docs[col][train_index])
        input_list_val.append(padded_docs[col][validation_index])
        input_list_test.append(padded_docs[col][data.shape[0]:])
    input_list_train.append(train_continuous.iloc[train_index])
    input_list_val.append(train_continuous.iloc[validation_index])
    input_list_test.append(test_continuous)
    
    y_train = Delivery_Time.iloc[train_index].values
    y_val = Delivery_Time.iloc[validation_index].values # np.log(train[['price']].iloc[validation_index].values+1)
    
    model = create_model(cat_cols, num_cols, word_index, max_length)
    if fold_number == 1:
        print(model.summary())
    filepath=file_path+"/model.best.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    reduce_learning_rate = ReduceLROnPlateau(monitor = "val_accuracy", factor = 0.6, patience=2, min_lr=1e-6,  verbose=1, mode = 'max')
    early_stopping = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=10, restore_best_weights = True)
    callbacks_list = [checkpoint,reduce_learning_rate,early_stopping]
    entity_embedding_model = model.fit(input_list_train,y_train, 
                             validation_data=(input_list_val,y_val), 
                             epochs=100,#25
                             callbacks=callbacks_list, 
                             shuffle=False, 
                             batch_size=128, #64
                             verbose=1)
    
    # Predictions
    if np.max(entity_embedding_model.history['val_accuracy']) > 0.79:
        new_model = load_model(filepath, custom_objects={'Attention': Attention})
        predictions = new_model.predict(input_list_test,verbose=1)
        test_preds += predictions
        predict_test += 1
    
    K.clear_session()
    fold_number += 1

test_preds /= predict_test #splits

preds = pd.DataFrame(test_preds)
preds.columns = Delivery_Time.columns
preds.head()


preds = pd.DataFrame()
preds['10 minutes'] = lstm_preds['10 minutes']*0.3 + cnn_preds['10 minutes']*0.3 + gru_preds['10 minutes']*0.4
preds['20 minutes'] = lstm_preds['20 minutes']*0.3 + cnn_preds['20 minutes']*0.3 + gru_preds['20 minutes']*0.4
preds['30 minutes'] = lstm_preds['30 minutes']*0.3 + cnn_preds['30 minutes']*0.3 + gru_preds['30 minutes']*0.4
preds['45 minutes'] = lstm_preds['45 minutes']*0.3 + cnn_preds['45 minutes']*0.3 + gru_preds['45 minutes']*0.4
preds['65 minutes'] = lstm_preds['65 minutes']*0.3 + cnn_preds['65 minutes']*0.3 + gru_preds['65 minutes']*0.4
preds['80 minutes'] = lstm_preds['80 minutes']*0.3 + cnn_preds['80 minutes']*0.3 + gru_preds['80 minutes']*0.4
preds['120 minutes'] = lstm_preds['120 minutes']*0.3 + cnn_preds['120 minutes']*0.3 + gru_preds['120 minutes']*0.4

submission = pd.DataFrame()
submission['Delivery_Time'] = preds.idxmax(axis=1)
submission = submission['Delivery_Time']
submission.to_excel(wd + 'submission.xlsx', header = True, index=False)





num_cols = train_continuous.columns.to_list()
file_path = wd + "/model"
directory = os.path.dirname(file_path)
print(file_path)
try:
    os.stat(file_path)
except:
    os.mkdir(file_path) 
    
from glob import glob
glob('/model/*')

test_preds = np.zeros((test.shape[0],7))
input_list_train, input_list_val, input_list_test = [], [], []
for col in cat_cols:
    input_list_train.append(padded_docs[col][:11000])
    input_list_val.append(padded_docs[col][11000:data.shape[0]])
    input_list_test.append(padded_docs[col][data.shape[0]:])
input_list_train.append(train_continuous.iloc[:11000])
input_list_val.append(train_continuous.iloc[11000:data.shape[0]])
input_list_test.append(test_continuous)

y_train = Delivery_Time.iloc[:11000].values
y_val = Delivery_Time.iloc[11000:data.shape[0]].values

model = create_model(cat_cols, num_cols, word_index, max_length)
print(model.summary())
filepath=file_path+"/model.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
reduce_learning_rate = ReduceLROnPlateau(monitor = "val_accuracy", factor = 0.6, patience=4, min_lr=1e-6,  verbose=1, mode = 'max')
early_stopping = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=20)
callbacks_list = [checkpoint,reduce_learning_rate,early_stopping]
entity_embedding_model = model.fit(input_list_train,y_train, 
                         validation_data=(input_list_val,y_val), 
                         epochs=100,#25
                         callbacks=callbacks_list, 
                         shuffle=False, 
                         batch_size=64, #64
                         verbose=1)

# Predictions
new_model = load_model(filepath)
predictions = new_model.predict(input_list_test,verbose=1)
test_preds = predictions

K.clear_session()

preds = pd.DataFrame(test_preds)
#Delivery_Time_Reverse_Lookup = ['10 minutes', '20 minutes', '30 minutes', '45 minutes', '65 minutes', '80 minutes', '120 minutes']
preds.columns = Delivery_Time.columns
preds.head()




error = pd.DataFrame(new_model.predict(input_list_val,verbose=1))
validation = data.iloc[validation_index].copy().reset_index(drop = True)
error.columns = Delivery_Time.columns
pred_val = pd.DataFrame(error.idxmax(axis=1), columns = ['Predictions'])
validation['Predictions'] = pred_val['Predictions']

error_df = validation[validation['Delivery_Time'] != validation['Predictions']]



# Test
from keras.models import model_from_json
entity_embedding_model = model.fit(input_list_train,y_train, 
                             validation_data=(input_list_val,y_val), 
                             epochs=100,#25
                             callbacks=callbacks_list, 
                             shuffle=False, 
                             batch_size=128, #64
                             verbose=1)

# Convert your existing model to JSON
saved_model = model.to_json()
saved_model = entity_embedding_model.model.to_json()
model.save('keras-model.h5')

from keras.models import load_model
new_model = load_model('keras-model.h5', custom_objects={'Attention': Attention})

# Read model   
new_model = model_from_json(saved_model, custom_objects={'Attention': Attention})
weights = entity_embedding_model.model.get_weights()
new_model.set_weights(weights)

predictions = new_model.predict(input_list_train,verbose=2, batch_size=4096)
predictions2 = entity_embedding_model.model.predict(input_list_train,verbose=2, batch_size=4096)


import os
filepath = os.path.join('keras-model.h5')
if not os.path.exists('keras-model.h5'): #determine whether it exists
     os.makedirs('keras-model.h5') #Create if it does not exist