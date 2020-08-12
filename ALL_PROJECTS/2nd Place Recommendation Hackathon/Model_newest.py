
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
    target_cols = [x[7:] for x in sub.columns[1:]]
    
    
#    if RunType.upper() == 'MODEL':
#        item_popularity = raw.item_id.value_counts()[:15].index.to_list()
#    else:
#        item_popularity = raw.item_id.value_counts()[:200].index.to_list()
#    target_cols = list(set(target_cols + item_popularity))
    
    data=raw_data[raw_data['item_id'].isin(target_cols)]
#    data= raw_data.copy()
    data.sort_values(['InvoiceDate'],ascending=False,inplace=True)
    data['rank']=data.groupby(['customer_id'])['InvoiceDate'].rank(ascending=0,method='dense')
    data.sort_values(['customer_id','InvoiceDate'],inplace=True)
    
    # DV Creation
    if RunType.upper() == 'MODEL':
        
        data1=data[data['rank']<=1]
        
        data1.loc[:,'Target']=1
        
        data1=data1[data1['item_id'].isin(target)]
        
        dv=data1.groupby(['customer_id','item_id']).agg({'Target':'count'}).unstack()
        
        for i in dv.columns:
            dv[i]=dv[i].apply(lambda x: 1 if x>0 else 0)
            
        dv
        
        dv.columns=dv.columns.map('_'.join)
        
        dv.reset_index(level=['customer_id'],inplace=True)
    
    
    if RunType.upper() == 'MODEL':
        # rd=raw_data['InvoiceDate'].max()-timedelta(days=30)
        data=data[data['rank']>1]
        
    r=data.groupby(['customer_id'],as_index=False)['InvoiceDate'].max().rename(columns={'InvoiceDate':'rd'})
    data=pd.merge(data,r,how='left',on=['customer_id'])
    data['rd1']=data['rd']-timedelta(days=1)
    data['rd7']=data['rd']-timedelta(days=7)
    data['rd30']=data['rd']-timedelta(days=30)
    data['rd60']=data['rd']-timedelta(days=60)
    data['rd90']=data['rd']-timedelta(days=90)
    data['rd360']=data['rd']-timedelta(days=360)
    data['rd720']=data['rd']-timedelta(days=720)
    
    len(data)
    data.reset_index(inplace=True,drop=['index'])
    
    # TFIDF Creation
    data['item_id_2'] = 'ITEM_' + data['item_id']
    agg_df = data.sort_values('InvoiceDate', ascending=False).groupby(['customer_id']).agg({
        'item_id_2': lambda s: ', '.join(s)
        }).reset_index()
    
    # Excluding other items
    data=data[data['item_id'].isin(target_cols)]
    
    def freq(dataset,reference_date,f,g,l):
        return dataset[dataset['InvoiceDate']>=reference_date].groupby(['customer_id','item_id'],as_index=False).agg({'transaction_id':'count','item_amt':'mean','item_qty':'sum'}).rename(columns={'transaction_id':f,'item_amt':g,'item_qty':l})
        
    f1=freq(data,data['rd1'],'F1','M1','Q1')
    f7=freq(data,data['rd7'],'F7','M7','Q7')
    f30=freq(data,data['rd30'],'F30','M30','Q30')
    f60=freq(data,data['rd60'],'F60','M60','Q60')
    f90=freq(data,data['rd90'],'F90','M90','Q90')
    f360=freq(data,data['rd360'],'F360','M360','Q360')
    f720=freq(data,data['rd720'],'F720','M720','Q720')
    
    def cust_freq(dataset,reference_date,f,g,l):
        return dataset[dataset['InvoiceDate']>=reference_date].groupby(['customer_id'],as_index=False).agg({'transaction_id':'count','item_amt':'mean','item_qty':'sum'}).rename(columns={'transaction_id':f,'item_amt':g,'item_qty':l})

    c1=cust_freq(data,data['rd1'],'CF1','CM1','CQ1')
    c7=cust_freq(data,data['rd7'],'CF7','CM7','CQ7')
    c30=cust_freq(data,data['rd30'],'CF30','CM30','CQ30')
    c60=cust_freq(data,data['rd60'],'CF60','CM60','CQ60')
    c90=cust_freq(data,data['rd90'],'CF90','CM90','CQ90')
    c360=cust_freq(data,data['rd360'],'CF360','CM360','CQ360')
    c720=cust_freq(data,data['rd720'],'CF720','CM720','CQ720')
    
    # Last 3 transaction average, stdev item level
    data['PrevInvoiceDate'] = data.sort_values(['InvoiceDate'],ascending=True).groupby(['customer_id','item_id'])['InvoiceDate'].shift(1)
    data['T2InvoiceDate'] = data.sort_values(['InvoiceDate'],ascending=True).groupby(['customer_id','item_id'])['InvoiceDate'].shift(2)
    data['T3InvoiceDate'] = data.sort_values(['InvoiceDate'],ascending=True).groupby(['customer_id','item_id'])['InvoiceDate'].shift(3)
    
    data['DayDiff'] = (data['InvoiceDate'] - data['PrevInvoiceDate']).dt.days
    data['DayDiff2'] = (data['InvoiceDate'] - data['T2InvoiceDate']).dt.days
    data['DayDiff3'] = (data['InvoiceDate'] - data['T3InvoiceDate']).dt.days
    
    tx_day_diff = data.groupby(['customer_id','item_id']).agg({'DayDiff': ['mean', 'std'], 'DayDiff2': ['mean', 'std'], 'DayDiff3': ['mean', 'std']}).reset_index()
    tx_day_diff.columns = ['customer_id', 'item_id', 'DayDiffMean', 'DayDiffStd', 'DayDiff2Mean', 'DayDiff2Std', 'DayDiff3Mean', 'DayDiff3Std']
    
    # Last 3 transaction average, stdev customer level
    data['C_PrevInvoiceDate'] = data.sort_values(['InvoiceDate'],ascending=True).groupby(['customer_id'])['InvoiceDate'].shift(1)
    data['C_T2InvoiceDate'] = data.sort_values(['InvoiceDate'],ascending=True).groupby(['customer_id'])['InvoiceDate'].shift(2)
    data['C_T3InvoiceDate'] = data.sort_values(['InvoiceDate'],ascending=True).groupby(['customer_id'])['InvoiceDate'].shift(3)
    
    data['C_DayDiff'] = (data['InvoiceDate'] - data['C_PrevInvoiceDate']).dt.days
    data['C_DayDiff2'] = (data['InvoiceDate'] - data['C_T2InvoiceDate']).dt.days
    data['C_DayDiff3'] = (data['InvoiceDate'] - data['C_T3InvoiceDate']).dt.days
    
    c_tx_day_diff = data.groupby(['customer_id']).agg({'C_DayDiff': ['mean', 'std'], 'C_DayDiff2': ['mean', 'std'], 'C_DayDiff3': ['mean', 'std']}).reset_index()
    c_tx_day_diff.columns = ['customer_id', 'C_DayDiffMean', 'C_DayDiffStd', 'C_DayDiff2Mean', 'C_DayDiff2Std', 'C_DayDiff3Mean', 'C_DayDiff3Std']

    
    f=pd.merge(pd.merge(pd.merge(pd.merge(pd.merge(pd.merge(f1,f7,how='outer',on=['customer_id','item_id']),f30,how='outer',on=['customer_id','item_id']),f60,how='outer',on=['customer_id','item_id']),f90,how='outer',on=['customer_id','item_id']),f360,how='outer',on=['customer_id','item_id']),f720,how='outer',on=['customer_id','item_id'])
    f = pd.merge(f, tx_day_diff, how='outer', on=['customer_id','item_id'])
    
    c=pd.merge(pd.merge(pd.merge(pd.merge(pd.merge(pd.merge(c1,c7,how='outer',on=['customer_id']),c30,how='outer',on=['customer_id']),c60,how='outer',on=['customer_id']),c90,how='outer',on=['customer_id']),c360,how='outer',on=['customer_id']),c720,how='outer',on=['customer_id'])

    cids = pd.DataFrame(raw_data.customer_id.unique(), columns=['customer_id'])
    len(cids.customer_id.unique())
    len(data.customer_id.unique())
    len(f.customer_id.unique())
    
#    c_r_l=data.groupby(['customer_id','item_id'],as_index=False).agg({'InvoiceDate':[lambda x: (rd-x.max()).days]})
    c_r_f=data.groupby(['customer_id','item_id'],as_index=False).agg({'InvoiceDate':[lambda x: (x.max()-x.min()).days]})
    
#    c_r_l.columns=c_r_l.columns.map('_'.join).str.strip('_')
    c_r_f.columns=c_r_f.columns.map('_'.join).str.strip('_')
    
#    c_r_l.rename(columns={'InvoiceDate_<lambda>':'R_L'},inplace=True)
    c_r_f.rename(columns={'InvoiceDate_<lambda>':'R_F'},inplace=True)
    
#    c_T_L=data.groupby(['customer_id'],as_index=False).agg({'InvoiceDate':[lambda x: (rd-x.max()).days]})
    c_T_F=data.groupby(['customer_id'],as_index=False).agg({'InvoiceDate':[lambda x: (x.max()-x.min()).days]})
    
#    c_T_L.columns=c_T_L.columns.map('_'.join).str.strip('_')
    c_T_F.columns=c_T_F.columns.map('_'.join).str.strip('_')
    
#    c_T_L.rename(columns={'InvoiceDate_<lambda>':'T_L'},inplace=True)
    c_T_F.rename(columns={'InvoiceDate_<lambda>':'T_F'},inplace=True)
    
    d=pd.merge(f,c_r_f,how='left',on=['customer_id','item_id'])   
    d=d.set_index(['customer_id','item_id']).unstack()    
    d.columns=d.columns.map('_'.join).str.strip('_')
    
    d.reset_index(level=['customer_id'],inplace=True)
    
    data_processed = pd.merge(pd.merge(pd.merge(d,c_T_F,how='left',on='customer_id'),c,how='left',on=['customer_id']),c_tx_day_diff,how='left',on=['customer_id'])
    
    data_processed=pd.merge(pd.DataFrame(raw_data['customer_id'].unique(), columns=['customer_id']),data_processed,how='left',on=['customer_id'])
    data_processed.fillna(999 ,inplace=True)
        
    data_processed = pd.merge(data_processed, agg_df, how='outer', on=['customer_id']).fillna('')

    
    if RunType.upper() == 'MODEL':
        data_processed = pd.merge(data_processed, dv, how='inner',on=['customer_id'])
        data_processed.fillna(0, inplace=True)
    
    print(data_processed.head())
    
    return data_processed

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

#raw.sort_values(['InvoiceDate'],ascending=False,inplace=True)
#raw['rank']=raw.groupby(['customer_id'])['InvoiceDate'].rank(ascending=0,method='dense')
#raw=raw[raw['rank']>1]
#raw.drop("rank", axis=1, inplace=True)
#univ_5 = generate_universe(raw, RunType='Model')
#
#raw.sort_values(['InvoiceDate'],ascending=False,inplace=True)
#raw['rank']=raw.groupby(['customer_id'])['InvoiceDate'].rank(ascending=0,method='dense')
#raw=raw[raw['rank']>1]
#raw.drop("rank", axis=1, inplace=True)
#univ_6 = generate_universe(raw, RunType='Model')

cols_to_keep = univ_1.columns.intersection(univ_2.columns).intersection(univ_3.columns).intersection(univ_4.columns)#.intersection(univ_5.columns).intersection(univ_6.columns)
univ_1 = univ_1[cols_to_keep]
univ_2 = univ_2[cols_to_keep]
univ_3 = univ_3[cols_to_keep]
univ_4 = univ_4[cols_to_keep]
#univ_5 = univ_5[cols_to_keep]
#univ_6 = univ_6[cols_to_keep]

data = pd.concat([univ_1, univ_2, univ_3, univ_4], axis=0)
#data = data[data.columns[data.isnull().mean() < 0.95]]
data.fillna(999 ,inplace=True)
del univ_1, univ_2, univ_3, univ_4#, univ_5, univ_6
gc.collect()

vectorizer = TfidfVectorizer()
tf_idf = vectorizer.fit_transform(data.item_id_2.values)
tf_idf_df = pd.DataFrame(tf_idf.toarray(), columns=vectorizer.get_feature_names())
tf_idf_df = tf_idf_df.add_prefix('tfidf_')
data.drop('item_id_2', axis=1, inplace=True)

data.reset_index(drop=True, inplace=True)
tf_idf_df.reset_index(drop=True, inplace=True)
data = pd.concat([data, tf_idf_df], axis=1)

data.to_csv("DL_MDL_UNIV.csv", header=True, index=False)

# Scoring Univ
raw = pd.read_csv('train.csv')
sub = pd.read_csv('submission.csv')
raw['InvoiceDate']=pd.to_datetime(raw['InvoiceDate'])
raw.dropna(inplace=True)
raw.head()

test = generate_universe(raw, RunType='Score')

tf_idf = vectorizer.transform(test.item_id_2.values)
tf_idf_df = pd.DataFrame(tf_idf.toarray(), columns=vectorizer.get_feature_names())
tf_idf_df = tf_idf_df.add_prefix('tfidf_')
test.drop('item_id_2', axis=1, inplace=True)

test.reset_index(drop=True, inplace=True)
tf_idf_df.reset_index(drop=True, inplace=True)
test = pd.concat([test, tf_idf_df], axis=1)

cols_to_keep = [col for col in data.columns if 'Target' not in col]
test = test[cols_to_keep]
test = test[test['customer_id'].isin(sub.customer_id)]
test.fillna(999, inplace=True)
test.to_csv("DL_MDL_UNIV_SCORE.csv", header=True, index=False)

data = data.drop_duplicates(subset=data.columns.difference(['customer_id']))







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

data.drop("customer_id", axis=1, inplace=True)
dependent_variables = [col for col in data.columns if 'Target' in col]

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

parameters = {
    'n_estimators': [150],
    'max_depth': [8],
    'max_features': ["auto", "log2", 30],
    'n_jobs': [-1],
    'oob_score': [True],
    'criterion': ["entropy"],
    'warm_start': [True],
    'bootstrap': [True]
    }

test_preds = pd.DataFrame(test['customer_id'])
for n, col in enumerate(dependent_variables):
    print(n, col)
    
    y_train = data[[col]].values.ravel()
#    y_val = validation[[col]].values
    
    X_train_matrix = csr_matrix(X_train, dtype='float32')
#    X_val_matrix = csr_matrix(X_val, dtype='float32')
    X_test_matrix = csr_matrix(X_test, dtype='float32')
    
#    selector = SelectFromModel(estimator=LogisticRegression()).fit(X_train_matrix, y_train)
#    X_train_matrix = selector.transform(X_train_matrix)
#    X_test_matrix = selector.transform(X_test_matrix)
    
    LR = RandomForestClassifier()
    clf = GridSearchCV(LR, parameters)
    clf.fit(X_train_matrix, y_train)
    best_parameters = clf.best_params_
    best_score = clf.best_score_
    
#    model_lr = RandomForestClassifier(**best_parameters)
#    model_lr = RandomForestClassifier(n_estimators=1000,max_depth=8,max_features=150,\
#                                       n_jobs=-1,oob_score=True,\
#                                       random_state=50,criterion="entropy",\
#                                           warm_start=True,bootstrap=True)
#    model_lr.fit(X_train_matrix, y_train) 
    roc=roc_auc_score(y_train, clf.predict_proba(X_train_matrix)[:,1])
    print('Best Score: ', best_score)
    print('Best Params', best_parameters)
#    roc=roc_auc_score(y_val, model_lr.predict_proba(X_val_matrix)[:,1])
#    print('Val AUC: ', roc)
    
    test_preds[col] = clf.predict_proba(X_test_matrix)[:,1]
    
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
mods.to_csv(wd + "predictions_KB_NEW_1.csv", index=False)




# Logistic Regression

# RFM
def build_model(data, test, exclude_cols):
    
    def warn(*args, **kwargs):
        pass
    import warnings
    warnings.warn = warn
    
    if "customer_id" in data.columns:
        data.drop("customer_id", axis=1, inplace=True)
    dependent_variables = [col for col in data.columns if 'Target' in col]
    
    parameters = {
    'solver': ['lbfgs'],
    'C': [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01],
    'l1_ratio': [0.001, 0.01]
    }

    
    test_preds = pd.DataFrame(test['customer_id'])
    for n, col in enumerate(dependent_variables):
        print(n, col)
        
        # Downsampling        
        count_class_0, count_class_1 = data[col].value_counts()
        df_class_0 = data[data[col] == 0]
        df_class_1 = data[data[col] == 1]        
        df_class_0_under = df_class_0.sample(count_class_1)
        df_under = pd.concat([df_class_0_under, df_class_1], axis=0)        
        print('Random under-sampling:')
        print(df_under[col].value_counts())
        
        X_train = df_under[[col for col in df_under.columns if col not in dependent_variables and col not in exclude_cols]].values
        
        X_test = test[[col for col in test.columns if 'customer_id' not in col and col not in exclude_cols]].values
        print(X_train.shape)
        
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        
        y_train = df_under[[col]].values
        
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
    return mods

#RFM Features Only
exclude_cols = [col for col in data.columns if 'tfidf' in col.lower()]
rfm = build_model(data, test, exclude_cols)

# TFIDF all items
exclude_cols = [col for col in data.columns if 'tfidf' not in col.lower()]
tfidf_all = build_model(data, test, exclude_cols)

# RFM + TFIDF target items
target = [col.split('Target_')[1] for col in dependent_variables]
tfidf_non_target_variables = [col for col in data.columns if 'tfidf_item_' in col]
tfidf_non_target_variables = [col for col in tfidf_non_target_variables if col.split('tfidf_item_')[1] not in target]
exclude_cols = [col for col in data.columns if col in tfidf_non_target_variables]
tfidf_target = build_model(data, test, exclude_cols)

# RFM + TFIDF all items
exclude_cols = []
rfm_tfidf = build_model(data, test, exclude_cols)
rfm_tfidf.to_csv(wd + "predictions_KB_NEW_L1.csv", index=False)

