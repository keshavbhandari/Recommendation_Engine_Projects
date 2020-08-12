# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 22:44:50 2020

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

def generate_universe(raw_data, RunType='Model'):
    sub=pd.read_csv('submission.csv')
    target=[x[7:] for x in sub.columns[1:]]
    
    #if RunType.upper() == 'MODEL':
    #    rd=raw['InvoiceDate'].max()-timedelta(days=30)
    #else:
    #    rd=raw['InvoiceDate'].max()
        
    if RunType.upper() == 'MODEL':
        raw.sort_values(['InvoiceDate'],ascending=False,inplace=True)
        raw['rank']=raw.groupby(['customer_id'])['InvoiceDate'].rank(ascending=0,method='dense')

        # DV Universe
        dv = raw[(raw['item_id'].isin(target)) & (raw['rank']<=1) & (raw['item_qty']>0)]
        dv.drop('rank', axis=1, inplace=True)
        dv.loc[:, 'Target'] = 1
        
        dv=dv.groupby(['customer_id','item_id']).agg({'Target':'count'}).unstack()
        for i in dv.columns:
            dv[i]=dv[i].apply(lambda x: 1 if x>0 else 0)
        dv.columns = dv.columns.get_level_values(1)
        dv.reset_index(level=['customer_id'],inplace=True)
        
        dv_df = pd.melt(dv, id_vars =['customer_id'], value_vars = target, var_name='item_id', value_name="Target")
    
    if RunType.upper() == 'MODEL':
        # IV Universe
        iv_full = raw[raw['rank']>1]
        iv_full.drop('rank', axis=1, inplace=True)
    else:
        iv_full = raw.copy()
        
    # Net Price    
    iv_full['Net_Price'] = iv_full['item_amt'] * iv_full['item_qty'] 
        
    # DL Feature Creation
    iv_full['categorical_sequence_items'] = 'ITEM_' + iv_full['item_id']
    agg_df = iv_full.sort_values('InvoiceDate', ascending=False).groupby(['customer_id']).agg({
        'categorical_sequence_items': lambda s: ', '.join(s)
        }).reset_index()
    
    impp = iv_full.groupby(['customer_id', 'item_id'],as_index=False)['transaction_id'].count().rename(columns={'transaction_id': 'count'}).sort_values(['count'], ascending=False)
    impp['rank'] = impp.groupby(['customer_id'], as_index=False)['count'].rank(ascending=0,method='dense')
    impp = impp[impp['rank']==1].drop(['rank'],axis=1)
    impp.drop_duplicates(subset=['customer_id'], keep='first', inplace=True)
    impp.columns = ['customer_id', 'IMPP', 'IMPP_count']
    
    # Reference Dates
    r=iv_full.groupby(['customer_id'],as_index=False)['InvoiceDate'].max().rename(columns={'InvoiceDate':'rd'})
    iv_full=pd.merge(iv_full,r,how='left',on=['customer_id'])
    iv_full['rd1']=iv_full['rd']-timedelta(days=1)
    iv_full['rd7']=iv_full['rd']-timedelta(days=7)
    iv_full['rd30']=iv_full['rd']-timedelta(days=30)
    iv_full['rd60']=iv_full['rd']-timedelta(days=60)
    iv_full['rd90']=iv_full['rd']-timedelta(days=90)
    iv_full['rd360']=iv_full['rd']-timedelta(days=360)
    iv_full['rd720']=iv_full['rd']-timedelta(days=720)
    
    def cust_freq(dataset,reference_date,f,g,l,k):
        return dataset[dataset['InvoiceDate']>=reference_date].groupby(['customer_id'],as_index=False).agg({'transaction_id':'count','item_amt':'mean','item_qty':'sum','Net_Price': 'sum'}).rename(columns={'transaction_id':f,'item_amt':g,'item_qty':l,'Net_Price':k})
    
    c1=cust_freq(iv_full,iv_full['rd1'],'CF1','CM1','CQ1','CN1')
    c7=cust_freq(iv_full,iv_full['rd7'],'CF7','CM7','CQ7','CN7')
    c30=cust_freq(iv_full,iv_full['rd30'],'CF30','CM30','CQ30','CN30')
    c60=cust_freq(iv_full,iv_full['rd60'],'CF60','CM60','CQ60','CN60')
    c90=cust_freq(iv_full,iv_full['rd90'],'CF90','CM90','CQ90','CN90')
    c360=cust_freq(iv_full,iv_full['rd360'],'CF360','CM360','CQ360','CN360')
    c720=cust_freq(iv_full,iv_full['rd720'],'CF720','CM720','CQ720','CN720')
    
    # Last 3 transaction average, stdev customer level
    iv_full['C_PrevInvoiceDate'] = iv_full.sort_values(['InvoiceDate'],ascending=True).groupby(['customer_id'])['InvoiceDate'].shift(1)
    iv_full['C_T2InvoiceDate'] = iv_full.sort_values(['InvoiceDate'],ascending=True).groupby(['customer_id'])['InvoiceDate'].shift(2)
    iv_full['C_T3InvoiceDate'] = iv_full.sort_values(['InvoiceDate'],ascending=True).groupby(['customer_id'])['InvoiceDate'].shift(3)
    
    iv_full['C_DayDiff'] = (iv_full['InvoiceDate'] - iv_full['C_PrevInvoiceDate']).dt.days
    iv_full['C_DayDiff2'] = (iv_full['InvoiceDate'] - iv_full['C_T2InvoiceDate']).dt.days
    iv_full['C_DayDiff3'] = (iv_full['InvoiceDate'] - iv_full['C_T3InvoiceDate']).dt.days
    
    c_tx_day_diff = iv_full.groupby(['customer_id']).agg({'C_DayDiff': ['mean', 'std'], 'C_DayDiff2': ['mean', 'std'], 'C_DayDiff3': ['mean', 'std']}).reset_index()
    c_tx_day_diff.columns = ['customer_id', 'C_DayDiffMean', 'C_DayDiffStd', 'C_DayDiff2Mean', 'C_DayDiff2Std', 'C_DayDiff3Mean', 'C_DayDiff3Std']
    
    # Exclude Non Target Items
    iv_full = iv_full[(iv_full['item_id'].isin(target))]
    
    # TFIDF Feature Creation
    iv_full['tfidf_item'] = 'ITEM_' + iv_full['item_id']
    agg_df_2 = iv_full.sort_values('InvoiceDate', ascending=False).groupby(['customer_id']).agg({
        'tfidf_item': lambda s: ', '.join(s)
        }).reset_index()
    
    # Reference Dates
    iv_full.drop('rd', axis=1, inplace=True)
    r=iv_full.groupby(['customer_id'],as_index=False)['InvoiceDate'].max().rename(columns={'InvoiceDate':'rd'})
    iv_full=pd.merge(iv_full,r,how='left',on=['customer_id'])
    iv_full['rd1']=iv_full['rd']-timedelta(days=1)
    iv_full['rd7']=iv_full['rd']-timedelta(days=7)
    iv_full['rd30']=iv_full['rd']-timedelta(days=30)
    iv_full['rd60']=iv_full['rd']-timedelta(days=60)
    iv_full['rd90']=iv_full['rd']-timedelta(days=90)
    iv_full['rd360']=iv_full['rd']-timedelta(days=360)
    iv_full['rd720']=iv_full['rd']-timedelta(days=720)
       
    def freq(dataset,reference_date,f,g,l,k):
        return dataset[dataset['InvoiceDate']>=reference_date].groupby(['customer_id','item_id'],as_index=False).agg({'transaction_id':'count','item_amt':'sum','item_qty':'sum', 'Net_Price': 'sum'}).rename(columns={'transaction_id':f,'item_amt':g,'item_qty':l, 'Net_Price':k})
            
    f1=freq(iv_full,iv_full['rd1'],'F1','M1','Q1','N1')
    f7=freq(iv_full,iv_full['rd7'],'F7','M7','Q7','N7')
    f30=freq(iv_full,iv_full['rd30'],'F30','M30','Q30','N30')
    f60=freq(iv_full,iv_full['rd60'],'F60','M60','Q60','N60')
    f90=freq(iv_full,iv_full['rd90'],'F90','M90','Q90','N90')
    f360=freq(iv_full,iv_full['rd360'],'F360','M360','Q360','N360')
    f720=freq(iv_full,iv_full['rd720'],'F720','M720','Q720','N720')    
    
    # Last 3 transaction average, stdev item level
    iv_full['PrevInvoiceDate'] = iv_full.sort_values(['InvoiceDate'],ascending=True).groupby(['customer_id','item_id'])['InvoiceDate'].shift(1)
    iv_full['T2InvoiceDate'] = iv_full.sort_values(['InvoiceDate'],ascending=True).groupby(['customer_id','item_id'])['InvoiceDate'].shift(2)
    iv_full['T3InvoiceDate'] = iv_full.sort_values(['InvoiceDate'],ascending=True).groupby(['customer_id','item_id'])['InvoiceDate'].shift(3)
    
    iv_full['DayDiff'] = (iv_full['InvoiceDate'] - iv_full['PrevInvoiceDate']).dt.days
    iv_full['DayDiff2'] = (iv_full['InvoiceDate'] - iv_full['T2InvoiceDate']).dt.days
    iv_full['DayDiff3'] = (iv_full['InvoiceDate'] - iv_full['T3InvoiceDate']).dt.days
    
    tx_day_diff = iv_full.groupby(['customer_id','item_id']).agg({'DayDiff': ['mean', 'std'], 'DayDiff2': ['mean', 'std'], 'DayDiff3': ['mean', 'std']}).reset_index()
    tx_day_diff.columns = ['customer_id', 'item_id', 'DayDiffMean', 'DayDiffStd', 'DayDiff2Mean', 'DayDiff2Std', 'DayDiff3Mean', 'DayDiff3Std']    
    
    
    
    f=pd.merge(pd.merge(pd.merge(pd.merge(pd.merge(pd.merge(f1,f7,how='outer',on=['customer_id','item_id']),f30,how='outer',on=['customer_id','item_id']),f60,how='outer',on=['customer_id','item_id']),f90,how='outer',on=['customer_id','item_id']),f360,how='outer',on=['customer_id','item_id']),f720,how='outer',on=['customer_id','item_id'])
    f = pd.merge(f, tx_day_diff, how='outer', on=['customer_id','item_id'])
    
    c=pd.merge(pd.merge(pd.merge(pd.merge(pd.merge(pd.merge(c1,c7,how='outer',on=['customer_id']),c30,how='outer',on=['customer_id']),c60,how='outer',on=['customer_id']),c90,how='outer',on=['customer_id']),c360,how='outer',on=['customer_id']),c720,how='outer',on=['customer_id'])
    
#    cids = pd.DataFrame(raw.customer_id.unique(), columns=['customer_id'])
    
    #    c_r_l=data.groupby(['customer_id','item_id'],as_index=False).agg({'InvoiceDate':[lambda x: (rd-x.max()).days]})
    c_r_f = iv_full.groupby(['customer_id','item_id'],as_index=False).agg({'InvoiceDate':[lambda x: (x.max()-x.min()).days]})
    
    #    c_r_l.columns=c_r_l.columns.map('_'.join).str.strip('_')
    c_r_f.columns=c_r_f.columns.map('_'.join).str.strip('_')
    
    #    c_r_l.rename(columns={'InvoiceDate_<lambda>':'R_L'},inplace=True)
    c_r_f.rename(columns={'InvoiceDate_<lambda>':'R_F'},inplace=True)
    
    #    c_T_L=data.groupby(['customer_id'],as_index=False).agg({'InvoiceDate':[lambda x: (rd-x.max()).days]})
    c_T_F = iv_full.groupby(['customer_id'],as_index=False).agg({'InvoiceDate':[lambda x: (x.max()-x.min()).days]})
    
    #    c_T_L.columns=c_T_L.columns.map('_'.join).str.strip('_')
    c_T_F.columns=c_T_F.columns.map('_'.join).str.strip('_')
    
    #    c_T_L.rename(columns={'InvoiceDate_<lambda>':'T_L'},inplace=True)
    c_T_F.rename(columns={'InvoiceDate_<lambda>':'T_F'},inplace=True)
    
    iv_df_ci = pd.merge(f,c_r_f,how='outer',on=['customer_id','item_id']) 
    iv_df_ci.fillna(999 ,inplace=True)
    
    iv_df_c = pd.merge(pd.merge(c_T_F, c,how='outer',on=['customer_id']),c_tx_day_diff,how='outer',on=['customer_id'])    
    iv_df_c = pd.merge(pd.merge(pd.DataFrame(raw['customer_id'].unique(), columns=['customer_id']),iv_df_c,how='outer',on=['customer_id']), impp, how="outer", on=['customer_id'])
    iv_df_c.fillna(999 ,inplace=True)
        
    iv_df_c = pd.merge(pd.merge(iv_df_c, agg_df, how='outer', on=['customer_id']), agg_df_2, how='outer', on=['customer_id']).fillna('')
    
    
    if RunType.upper() == 'MODEL':
        iv_df = pd.merge(iv_df_ci, dv_df, how='right',on=['customer_id', 'item_id'])
        iv_df.fillna(999, inplace=True)
        iv_df = pd.merge(iv_df, iv_df_c, how='left',on=['customer_id'])
        iv_df['categorical_sequence_items'].fillna('' ,inplace=True)
        iv_df.fillna(999 ,inplace=True)
    else:
        # Cross Join
        items = pd.DataFrame(iv_df_ci.item_id.unique(), columns=['item_id'])
        cids = pd.DataFrame(iv_df_ci.customer_id.unique(), columns=['customer_id'])
        items.loc[:,'constant'] = 1
        cids.loc[:,'constant'] = 1
        item_cid_combinations = pd.merge(cids, items, how='outer', on=['constant'])
        iv_df_ci = pd.merge(iv_df_ci, item_cid_combinations, how='outer', on=['customer_id', 'item_id'])
        iv_df = pd.merge(iv_df_ci, iv_df_c, how='left',on=['customer_id'])
        iv_df['categorical_sequence_items'].fillna('' ,inplace=True)
        iv_df.fillna(999 ,inplace=True)
        
    print(iv_df.head())
    return iv_df


univ_1 = generate_universe(raw, RunType='Model')

raw.sort_values(['InvoiceDate'],ascending=False,inplace=True)
raw['rank']=raw.groupby(['customer_id'])['InvoiceDate'].rank(ascending=0,method='dense')
raw=raw[raw['rank']>1]
raw.drop("rank", axis=1, inplace=True)
univ_2= generate_universe(raw, RunType='Model')

raw.sort_values(['InvoiceDate'],ascending=False,inplace=True)
raw['rank']=raw.groupby(['customer_id'])['InvoiceDate'].rank(ascending=0,method='dense')
raw=raw[raw['rank']>1]
raw.drop("rank", axis=1, inplace=True)
univ_3 = generate_universe(raw, RunType='Model')

raw.sort_values(['InvoiceDate'],ascending=False,inplace=True)
raw['rank']=raw.groupby(['customer_id'])['InvoiceDate'].rank(ascending=0,method='dense')
raw=raw[raw['rank']>1]
raw.drop("rank", axis=1, inplace=True)
univ_4 = generate_universe(raw, RunType='Model')

raw.sort_values(['InvoiceDate'],ascending=False,inplace=True)
raw['rank']=raw.groupby(['customer_id'])['InvoiceDate'].rank(ascending=0,method='dense')
raw=raw[raw['rank']>1]
raw.drop("rank", axis=1, inplace=True)
univ_5 = generate_universe(raw, RunType='Model')

raw.sort_values(['InvoiceDate'],ascending=False,inplace=True)
raw['rank']=raw.groupby(['customer_id'])['InvoiceDate'].rank(ascending=0,method='dense')
raw=raw[raw['rank']>1]
raw.drop("rank", axis=1, inplace=True)
univ_6 = generate_universe(raw, RunType='Model')

cols_to_keep = univ_1.columns.intersection(univ_2.columns).intersection(univ_3.columns).intersection(univ_4.columns)#.intersection(univ_5.columns).intersection(univ_6.columns)
univ_1 = univ_1[cols_to_keep]
univ_2 = univ_2[cols_to_keep]
univ_3 = univ_3[cols_to_keep]
univ_4 = univ_4[cols_to_keep]
univ_5 = univ_5[cols_to_keep]
univ_6 = univ_6[cols_to_keep]

data = pd.concat([univ_1, univ_2, univ_3, univ_4, univ_5, univ_6], axis=0)
#data = data[data.columns[data.isnull().mean() < 0.95]]
data.fillna(999 ,inplace=True)
del univ_1, univ_2, univ_3, univ_4, univ_5, univ_6
gc.collect()

vectorizer = TfidfVectorizer()
tf_idf = vectorizer.fit_transform(data.tfidf_item.values)
tf_idf_df = pd.DataFrame(tf_idf.toarray(), columns=vectorizer.get_feature_names())
tf_idf_df = tf_idf_df.add_prefix('tfidf_')
data.drop('tfidf_item', axis=1, inplace=True)

data.reset_index(drop=True, inplace=True)
tf_idf_df.reset_index(drop=True, inplace=True)
data = pd.concat([data, tf_idf_df], axis=1)

# Don't use this line for deep learning
#items = pd.get_dummies(data['item_id'], prefix='Item_')
#data = pd.concat([data, items], axis=1)

data.to_csv("DL_MDL_UNIV.csv", header=True, index=False)

# Scoring Univ
raw = pd.read_csv('train.csv')
sub = pd.read_csv('submission.csv')
raw['InvoiceDate']=pd.to_datetime(raw['InvoiceDate'])
raw.dropna(inplace=True)
raw.head()

test = generate_universe(raw, RunType='Score')

tf_idf = vectorizer.transform(test.tfidf_item.values)
tf_idf_df = pd.DataFrame(tf_idf.toarray(), columns=vectorizer.get_feature_names())
tf_idf_df = tf_idf_df.add_prefix('tfidf_')
test.drop('tfidf_item', axis=1, inplace=True)

test.reset_index(drop=True, inplace=True)
tf_idf_df.reset_index(drop=True, inplace=True)
test = pd.concat([test, tf_idf_df], axis=1)

# Don't use this line for deep learning
#items = pd.get_dummies(test['item_id'], prefix='Item_')
#test = pd.concat([test, items], axis=1)

cols_to_keep = [col for col in data.columns if 'Target' not in col]
test = test[cols_to_keep]
test = test[test['customer_id'].isin(sub.customer_id)]
test.fillna(999, inplace=True)
test.to_csv("DL_MDL_UNIV_SCORE.csv", header=True, index=False)


a = raw.groupby(['customer_id', 'item_id'],as_index=False)['transaction_id'].count().rename(columns={'transaction_id': 'count'}).sort_values(['count'], ascending=False)
a['rank'] = a.groupby(['customer_id'], as_index=False)['count'].rank(ascending=0,method='dense')
a = a[a['rank']==1].drop(['rank'],axis=1)
a.drop_duplicates(subset=['customer_id'], keep='first', inplace=True)


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
from sklearn.linear_model import LogisticRegressionCV

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


if "customer_id" in data.columns:
    data.drop("customer_id", axis=1, inplace=True)
if "item_id" in data.columns:
    data_item_id = pd.DataFrame(data['item_id'])
    test_item_id = pd.DataFrame(test['item_id'])
    data.drop("item_id", axis=1, inplace=True)
    
dependent_variables = [col for col in data.columns if 'Target' in col]
    
parameters = {
'solver': ['lbfgs'],
'C': [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01],
'l1_ratio': [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
}

parameters = {
'solver': ['lbfgs'],
'C': [0.01],
'l1_ratio': [0.001]
}

test_preds = pd.DataFrame(test[['customer_id', 'item_id']])
for n, col in enumerate(dependent_variables):
    print(n, col)
    
#    # Downsampling        
#    count_class_0, count_class_1 = data[col].value_counts()
#    df_class_0 = data[data[col] == 0]
#    df_class_1 = data[data[col] == 1]        
#    df_class_0_under = df_class_0.sample(count_class_1)
#    df_under = pd.concat([df_class_0_under, df_class_1], axis=0)        
#    print('Random under-sampling:')
#    print(df_under[col].value_counts())
    
    X_train = data[[col for col in data.columns if col not in dependent_variables]].values
    
    X_test = test[[col for col in test.columns if 'customer_id' not in col and 'item_id' not in col]].values
    print(X_train.shape)
    
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    y_train = data[[col]].values
    
    X_train_matrix = csr_matrix(X_train, dtype='float32')
    X_test_matrix = csr_matrix(X_test, dtype='float32')
    
    selector = SelectFromModel(estimator=LogisticRegression()).fit(X_train_matrix, y_train)
    X_train_matrix = selector.transform(X_train_matrix)
    X_test_matrix = selector.transform(X_test_matrix)
    
    LR = LogisticRegression()
    clf = GridSearchCV(LR, parameters)
    clf.fit(X_train_matrix, y_train)
    best_parameters = clf.best_params_
    best_score = clf.best_score_

    print ("Best %s:" % (best_score))
    print ("Best parameters set:", best_parameters)
    
    model_lr = LogisticRegression(**best_parameters)
    model_lr.fit(X_train_matrix, y_train) 
    roc=roc_auc_score(y_train, model_lr.predict_proba(X_train_matrix)[:,1])
    print('Train AUC: ', roc)
    
    test_preds[col] = model_lr.predict_proba(X_test_matrix)[:,1]

