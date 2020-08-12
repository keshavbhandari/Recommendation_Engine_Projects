# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 14:51:57 2020

@author: kbhandari
"""

import pandas as pd
import numpy as np
from datetime import timedelta
import os
import gc
from sklearn.feature_extraction.text import TfidfVectorizer

RunType = 'Model'

wd = "C:/My Desktop/Hackathon/"
os.chdir(wd)

raw=pd.read_csv('train.csv')
raw['InvoiceDate']=pd.to_datetime(raw['InvoiceDate'])
raw.dropna(inplace=True)
raw.head()

# Class count
#count_class_0, count_class_1 = df_train.target.value_counts()
#
## Divide by class
#df_class_0 = df_train[df_train['target'] == 0]
#df_class_1 = df_train[df_train['target'] == 1]

#df_class_0_under = df_class_0.sample(count_class_1)
#df_test_under = pd.concat([df_class_0_under, df_class_1], axis=0)
#
#print('Random under-sampling:')
#print(df_test_under.target.value_counts())

def generate_universe(raw_data, RunType='Model'):
    
    sub=pd.read_csv('submission.csv')

    target=[x[7:] for x in sub.columns[1:]]
    
    if RunType.upper() == 'MODEL':
        item_popularity = raw.item_id.value_counts()[:30].index.to_list()
    else:
        item_popularity = raw.item_id.value_counts()[:200].index.to_list()
    new_target = list(set(target + item_popularity))
    
    data = raw_data.copy()
    
    if RunType.upper() == 'MODEL':
        rd=raw_data['InvoiceDate'].max()-timedelta(days=30)
    else:
        rd=raw_data['InvoiceDate'].max()
        
    if RunType.upper() == 'MODEL':
        data=data[data['InvoiceDate']<=rd]
        
    data.reset_index(inplace=True,drop=['index'])
    print(data.head())
    
    data = data[data['item_id'].isin(new_target)]
    
    # TFIDF Creation
    agg_df = data.sort_values('InvoiceDate', ascending=False).groupby(['customer_id']).agg({
        'item_id': lambda s: ', '.join(s)
        }).reset_index()
    
    # Last 3 transaction average, stdev
    data['PrevInvoiceDate'] = data.sort_values(['InvoiceDate'],ascending=True).groupby(['customer_id','item_id'])['InvoiceDate'].shift(1)
    data['T2InvoiceDate'] = data.sort_values(['InvoiceDate'],ascending=True).groupby(['customer_id','item_id'])['InvoiceDate'].shift(2)
    data['T3InvoiceDate'] = data.sort_values(['InvoiceDate'],ascending=True).groupby(['customer_id','item_id'])['InvoiceDate'].shift(3)
    
    data['DayDiff'] = (data['InvoiceDate'] - data['PrevInvoiceDate']).dt.days
    data['DayDiff2'] = (data['InvoiceDate'] - data['T2InvoiceDate']).dt.days
    data['DayDiff3'] = (data['InvoiceDate'] - data['T3InvoiceDate']).dt.days
    
    tx_day_diff = data.groupby(['customer_id','item_id']).agg({'DayDiff': ['mean', 'std'], 'DayDiff2': ['mean', 'std'], 'DayDiff3': ['mean', 'std']}).reset_index()
    tx_day_diff.columns = ['customer_id', 'item_id', 'DayDiffMean', 'DayDiffStd', 'DayDiff2Mean', 'DayDiff2Std', 'DayDiff3Mean', 'DayDiff3Std']

    tx_day_diff = tx_day_diff.set_index(['customer_id','item_id']).unstack()
    tx_day_diff.reset_index(level=['customer_id'],inplace=True)
    
    # DV Creation
    if RunType.upper() == 'MODEL':
        data1=raw_data[raw_data['InvoiceDate']>rd]
        
        data1.loc[:,'Target']=1
        
        data1=data1[data1['item_id'].isin(target)]
        
        dv=data1.groupby(['customer_id','item_id']).agg({'Target':'count'}).unstack()
        
        for i in dv.columns:
            dv[i]=dv[i].apply(lambda x: 1 if x>0 else 0)
        
        dv.columns=dv.columns.map('_'.join)
        dv.reset_index(level=['customer_id'],inplace=True)
        
    if RunType.upper() == 'MODEL':
        agg_df = pd.merge(agg_df, tx_day_diff, how='outer', on=['customer_id'])
        agg_df.fillna(999 ,inplace=True)
        data_processed = pd.merge(agg_df, dv, how='inner',on=['customer_id'])
        data_processed.fillna(0, inplace=True)
        print(data_processed.head())
        return data_processed
    
    else:
        agg_df = pd.merge(agg_df, tx_day_diff, how='outer', on=['customer_id'])
        agg_df.fillna(999 ,inplace=True)
        return agg_df
    
    
univ_1 = generate_universe(raw, RunType='Model')

rd=raw['InvoiceDate'].max()-timedelta(days=30)
raw=raw[raw['InvoiceDate']<=rd]
univ_2= generate_universe(raw, RunType='Model')

rd=raw['InvoiceDate'].max()-timedelta(days=30)
raw=raw[raw['InvoiceDate']<=rd]
univ_3 = generate_universe(raw, RunType='Model')

rd=raw['InvoiceDate'].max()-timedelta(days=30)
raw=raw[raw['InvoiceDate']<=rd]
univ_4 = generate_universe(raw, RunType='Model')

rd=raw['InvoiceDate'].max()-timedelta(days=30)
raw=raw[raw['InvoiceDate']<=rd]
univ_5 = generate_universe(raw, RunType='Model')

rd=raw['InvoiceDate'].max()-timedelta(days=30)
raw=raw[raw['InvoiceDate']<=rd]
univ_6 = generate_universe(raw, RunType='Model')

cols_to_keep = univ_1.columns.intersection(univ_2.columns).intersection(univ_3.columns).intersection(univ_4.columns).intersection(univ_5.columns).intersection(univ_6.columns)
univ_1 = univ_1[cols_to_keep]
univ_2 = univ_2[cols_to_keep]
univ_3 = univ_3[cols_to_keep]
univ_4 = univ_4[cols_to_keep]
univ_5 = univ_5[cols_to_keep]
univ_6 = univ_6[cols_to_keep]

data = pd.concat([univ_1, univ_2, univ_3, univ_4, univ_5, univ_6], axis=0)
del univ_1, univ_2, univ_3, univ_4, univ_5, univ_6
gc.collect()

vectorizer = TfidfVectorizer()
tf_idf = vectorizer.fit_transform(data.item_id.values)
tf_idf_df = pd.DataFrame(tf_idf.toarray(), columns=vectorizer.get_feature_names())
tf_idf_df = tf_idf_df.add_prefix('tfidf_')
data.drop('item_id', axis=1, inplace=True)

data.reset_index(drop=True, inplace=True)
tf_idf_df.reset_index(drop=True, inplace=True)
data = pd.concat([data, tf_idf_df], axis=1)
data.fillna(999 ,inplace=True)

# Scoring Univ
raw = pd.read_csv('train.csv')
sub = pd.read_csv('submission.csv')
raw['InvoiceDate']=pd.to_datetime(raw['InvoiceDate'])
raw.dropna(inplace=True)
raw.head()

test = generate_universe(raw, RunType='Score')
test = test[test['customer_id'].isin(sub.customer_id)]
cid = pd.DataFrame(sub[~sub['customer_id'].isin(test['customer_id'])]['customer_id'])
test = pd.merge(cid, test, how = "outer", on = ['customer_id'])
test.item_id.fillna(value = '', inplace=True)
test.fillna(999 ,inplace=True)

tf_idf = vectorizer.transform(test.item_id.values)
tf_idf_df = pd.DataFrame(tf_idf.toarray(), columns=vectorizer.get_feature_names())
tf_idf_df = tf_idf_df.add_prefix('tfidf_')
test.drop('item_id', axis=1, inplace=True)

test.reset_index(drop=True, inplace=True)
tf_idf_df.reset_index(drop=True, inplace=True)
test = pd.concat([test, tf_idf_df], axis=1)
cols_to_keep = [col for col in data.columns if 'Target' not in col]
test = test[cols_to_keep]





from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
from scipy.sparse import vstack, csr_matrix, save_npz, load_npz
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

data.drop("customer_id", axis=1, inplace=True)
dependent_variables = [col for col in data.columns if 'Target' in col]

parameters = {
    'solver': ['lbfgs'],
    'C': [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01, 0.001],
    'l1_ratio': [0.001, 0.01]
}

#parameters = {
#    'max_depth': [6,7,8,9,10],
#    'n_estimators': [500],
#    'oob_score': [True],
#    'n_jobs': [-1]
#}

#learn, validation = train_test_split(data, test_size = 0.1, random_state = 0)
X_train = data[[col for col in data.columns if col not in dependent_variables]].values
#X_train = learn[[col for col in learn.columns if col not in dependent_variables]].values
#X_val = validation[[col for col in validation.columns if col not in dependent_variables]].values

#sc = StandardScaler()
sc = MinMaxScaler()
X_train = sc.fit_transform(X_train)
#X_val = sc.transform(X_val)
X_test = test[[col for col in test.columns if 'customer_id' not in col]].values
X_test = sc.transform(X_test)

#pca = PCA(n_components=500)
#pca.fit(X_train)
#print("Variance: " ,np.sum(pca.explained_variance_ratio_))
#X_train = pca.transform(X_train)
#X_val = pca.transform(X_val)
#X_test = pca.transform(X_test)

test_preds = pd.DataFrame(test['customer_id'])
for n, col in enumerate(dependent_variables):
    print(n, col)
    
    y_train = data[[col]].values.ravel()
#    y_train = data[[col]].values
#    y_val = validation[[col]].values
    
    X_train_matrix = csr_matrix(X_train, dtype='float32')
#    X_val_matrix = csr_matrix(X_val, dtype='float32')
    X_test_matrix = csr_matrix(X_test, dtype='float32')
    
    selector = SelectFromModel(estimator=LogisticRegression()).fit(X_train_matrix, y_train)
    X_train_matrix = selector.transform(X_train_matrix)
    X_test_matrix = selector.transform(X_test_matrix)
    
#    LR = LogisticRegression()
#    LR = RandomForestClassifier()
#    clf = GridSearchCV(LR, parameters)
#    clf.fit(X_train_matrix, y_train)
#    best_parameters = clf.best_params_
#    best_score = clf.best_score_
#    
#    print ("Best %s:" % (best_score))
#    print ("Best parameters set:", best_parameters)
    
#    model_lr = LogisticRegression(**best_parameters)
    model_lr = RandomForestClassifier(n_estimators=1000,max_depth=8,\
                                       n_jobs=-1,oob_score=True,\
                                       random_state=50,criterion="entropy",\
                                           warm_start=True,bootstrap=True)
    model_lr.fit(X_train_matrix, y_train) 
    roc=roc_auc_score(y_train, model_lr.predict_proba(X_train_matrix)[:,1])
    print('Train AUC: ', roc)
#    roc=roc_auc_score(y_val, model_lr.predict_proba(X_val_matrix)[:,1])
#    print('Val AUC: ', roc)
    
    test_preds[col] = model_lr.predict_proba(X_test_matrix)[:,1]
    
sub = pd.read_csv('submission.csv')
popularity = dict(data[dependent_variables].mean())
popularity['Target_85123A'] = 1
popularity['Target_85099B'] = 0.9
popularity['Target_84879'] = 0.8

cid = sub[~sub['customer_id'].isin(test_preds['customer_id'])]
cid = cid[test_preds.columns]

for key, value in popularity.items():
    cid[key] = value

mods = pd.concat([test_preds, cid], axis=0)
mods = mods[mods['customer_id'].isin(sub['customer_id'])]
mods = mods[sub.columns]
mods = mods.sort_values(by="customer_id").reset_index(drop=True)
mods.to_csv(wd + "predictions_KB_123.csv", index=False)
