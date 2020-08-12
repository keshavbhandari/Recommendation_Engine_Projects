# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 18:51:53 2020

@author: kbhandari
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import gc
import pickle
from keras.models import Model
from keras.layers import Input, Reshape, Dot
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.layers import Add, Activation, Lambda
from keras.layers import Concatenate, Dense, Dropout
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,EarlyStopping,Callback
from keras.models import load_model, model_from_json

DV_WINDOW_START = '2011-09-09'
#DV_WINDOW_END = '2011-12-10'

wd = "C:/Users/kbhandari/OneDrive - Epsilon/Desktop/Not Just OLS/Recommendation_Engine/"
os.chdir(wd)
data = pd.read_csv(wd + "Data.csv")

data['Customer ID'] = data['Customer ID'].astype(str).str.extract('(\d+)', expand=False).dropna().astype(int).astype(str)
data['StockCode'] = data['StockCode'].str.extract('(\d+)', expand=False).dropna().astype(str)
data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])

holdout = data[data['InvoiceDate']>=DV_WINDOW_START]
data = data[data['InvoiceDate']<DV_WINDOW_START]

# Aggregate data
model_universe = data.groupby(['Customer ID', 'StockCode']).Invoice.agg('count').to_frame('COUNT_TXN').reset_index()
model_universe['COUNT_TXN'] = np.log(model_universe['COUNT_TXN']+1)
print(model_universe.describe())
print(model_universe.dtypes)

# Data labels




user_enc = LabelEncoder()
model_universe['USER_ID'] = user_enc.fit_transform(model_universe['Customer ID'].values)
n_users = model_universe['USER_ID'].nunique()
item_enc = LabelEncoder()
model_universe['SKU_ID'] = item_enc.fit_transform(model_universe['StockCode'].values)
n_category = model_universe['SKU_ID'].nunique()
model_universe['rating'] = model_universe['COUNT_TXN'].values.astype(np.float32)
min_rating = min(model_universe['rating'])
max_rating = max(model_universe['rating'])
print(n_users, n_category, min_rating, max_rating)

np.save('CID_encoder.npy', user_enc.classes_)
np.save('Category_encoder.npy', item_enc.classes_)


X = model_universe[['USER_ID', 'SKU_ID']].values
y = model_universe['rating'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
#del data, X, y
#gc.collect()


n_factors = 50
X_train_array = [X_train[:, 0], X_train[:, 1]]
X_test_array = [X_test[:, 0], X_test[:, 1]]




class EmbeddingLayer:
    def __init__(self, n_items, n_factors):
        self.n_items = n_items
        self.n_factors = n_factors
    
    def __call__(self, x):
        x = Embedding(self.n_items, self.n_factors, embeddings_initializer='he_normal',
                      embeddings_regularizer=l2(1e-6))(x)
        x = Reshape((self.n_factors,))(x)
        return x

def RecommenderNet(n_users, n_category, n_factors, min_rating, max_rating):
    user = Input(shape=(1,))
    u = EmbeddingLayer(n_users, n_factors)(user)
    
    category = Input(shape=(1,))
    m = EmbeddingLayer(n_category, n_factors)(category)
    
    x = Concatenate()([u, m])
    x = Dropout(0.05)(x)
    
    x = Dense(10, kernel_initializer='he_normal')(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    
    x = Dense(1, kernel_initializer='he_normal')(x)
    x = Activation('sigmoid')(x)
    x = Lambda(lambda x: x * (max_rating - min_rating) + min_rating)(x)
    model = Model(inputs=[user, category], outputs=x)
    opt = Adam(lr=0.001)
    model.compile(loss='mean_squared_error', optimizer=opt)
    return model


model = RecommenderNet(n_users, n_category, n_factors, min_rating, max_rating)
model.summary()

reduce_learning_rate = ReduceLROnPlateau(monitor = "val_loss", factor = 0.6, patience=1, min_lr=1e-5,  verbose=1, mode = 'min')
early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2, restore_best_weights = True)
callbacks_list = [reduce_learning_rate,early_stopping]

history = model.fit(x=X_train_array, y=y_train, 
                    batch_size=128, epochs=20,
                    verbose=1, validation_data=(X_test_array, y_test),
                    callbacks=callbacks_list)

# loss: 0.0941 - val_loss: 0.0924

# Scoring
architecture = history.model.to_json()
weights = history.model.get_weights()

pickle.dump(architecture, open(wd+"keras-model.json", "wb"))
pickle.dump(weights, open(wd+"keras-weights.pkl", "wb"))

loaded_model = pickle.load(open(wd+"keras-model.json", "rb"))
loaded_weights = pickle.load(open(wd+"keras-weights.pkl", "rb"))
new_model = model_from_json(loaded_model)
new_model.set_weights(loaded_weights)      



# Scoring
user_enc = LabelEncoder()
user_enc.classes_ = np.load('CID_encoder.npy', allow_pickle = True)
item_enc = LabelEncoder()
item_enc.classes_ = np.load('Category_encoder.npy', allow_pickle = True)

cid = pd.DataFrame({'CID': model_universe['USER_ID'].unique()})
pid = pd.DataFrame({'Category': model_universe['SKU_ID'].unique()})

chunk_size = 100
cid['Sequence'] = np.arange(len(cid))
cid['Sequence'] = cid['Sequence']/len(cid)
cid['Chunk'] = pd.cut(cid['Sequence'], bins=chunk_size, labels=False)
Chunks = cid['Chunk'].unique()
cid.drop(['Sequence'], axis=1, inplace=True)

first_time = True
for Chunk in Chunks:
    print("Processing chunk:", Chunk)
    if first_time:
        cid_array = cid[cid['Chunk']==Chunk]['CID'].values
        pid_array = pid.values
        tmp = pd.DataFrame(np.transpose([np.tile(cid_array, len(pid_array)), np.repeat(pid_array, len(cid_array))]), columns=['CID','Category'])
        tmp['predictions'] = new_model.predict([tmp['CID'].values, tmp['Category'].values],verbose=1, batch_size=8192)
        CID_SKU_Combinations = tmp.groupby(['CID']).apply(lambda x: x.nlargest(10,['predictions'])).reset_index(drop=True)
        first_time = False
    else:
        cid_array = cid[cid['Chunk']==Chunk]['CID'].values
        pid_array = pid.values
        tmp = pd.DataFrame(np.transpose([np.tile(cid_array, len(pid_array)), np.repeat(pid_array, len(cid_array))]), columns=['CID','Category'])
        tmp['predictions'] = new_model.predict([tmp['CID'].values, tmp['Category'].values],verbose=1, batch_size=8192)
        tmp = tmp.groupby(['CID']).apply(lambda x: x.nlargest(10,['predictions'])).reset_index(drop=True)
        CID_SKU_Combinations = CID_SKU_Combinations.append(tmp, ignore_index = True)


CID_SKU_Combinations['CID'] = user_enc.inverse_transform(CID_SKU_Combinations['CID'].values)
CID_SKU_Combinations['Category'] = item_enc.inverse_transform(CID_SKU_Combinations['Category'].values)
CID_SKU_Combinations["Rank"] = CID_SKU_Combinations.groupby("CID")["predictions"].rank('dense',ascending=False)
