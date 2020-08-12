# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 17:52:21 2019

@author: kbhandari
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
import xgboost as xgb
import lightgbm as lgb

wd = "C:/Users/kbhandari/OneDrive - Epsilon/Desktop/Not Just OLS/Food Delivery Time/Participants Data/"
data = pd.read_excel(wd + "Data_Train.xlsx", na_values = '-', encoding="utf-8")
test = pd.read_excel(wd + "Data_Test.xlsx", na_values = '-', encoding="utf-8")

def data_preprocess(data):
    data = data.replace('[^\w\s]','',regex=True)
    data['New'] = np.where(data['Rating']=='NEW', 1, 0)
    data['Opening_Soon'] = np.where(data['Rating']=='Opening Soon', 1, 0)
    data['Temporarily Closed'] = np.where(data['Rating']=='Temporarily Closed', 1, 0)
    data = data.replace('NEW', None)
    data = data.replace('Opening Soon', None)
    data = data.replace('Temporarily Closed', None)
    data['Average_Cost'] = np.where(data['Average_Cost']=='for', data['Average_Cost'].mode()[0], data['Average_Cost'])
    
    data[['Average_Cost', 'Minimum_Order', 'Rating']] = data[['Average_Cost', 'Minimum_Order', 'Rating']].apply(pd.to_numeric) 
    data = data.replace(np.nan, -1)
    return data

data = data_preprocess(data)
nrow_data = len(data)
test = data_preprocess(test)
#Delivery_Time = pd.get_dummies(pd.DataFrame(data['Delivery_Time']), prefix='', prefix_sep = '')
Delivery_Time_Lookup = {'10 minutes': 0, '120 minutes': 6, '20 minutes': 1, '30 minutes': 2, '45 minutes': 3, '65 minutes': 4, '80 minutes': 5}
Delivery_Time = data.replace({"Delivery_Time": Delivery_Time_Lookup})['Delivery_Time']

combine = data.append(test,sort=False).reset_index(drop=True)

tv = TfidfVectorizer(max_features=None,
                     ngram_range=(1, 3), min_df=2, token_pattern=r'(?u)\b\w+\b')
X_Location = tv.fit_transform(combine['Location'])
X_Cuisines = tv.fit_transform(combine['Cuisines'])
X_dummies = csr_matrix(pd.get_dummies(combine[['Average_Cost', 'Minimum_Order', 'Rating', 'Votes', 'Reviews', 'New', 'Opening_Soon', 'Temporarily Closed']], sparse=True).values)
sparse_combined = hstack((X_Location, X_Cuisines, X_dummies)).tocsr()


X = sparse_combined[:nrow_data]
X_test = sparse_combined[nrow_data:]
y = np.array(Delivery_Time)

model = Ridge(solver="auto", fit_intercept=True, random_state=369)
model.fit(X, y)
model.score(X, y)

test_preds = pd.DataFrame(model.predict(X=X_test))
test_preds.columns = Delivery_Time.columns
test_preds.head()


from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# XGBoost
dtrain = xgb.DMatrix(X_train,label=y_train)
dvalid = xgb.DMatrix(X_val, label=y_val)
param = {'max_depth':6, 'eta':0.01, 'silent':1, 'objective': 'multi:softprob', 'num_class': 7}
param['nthread'] = 4
param['eval_metric'] = 'mlogloss'
param['subsample'] = 0.7
param['colsample_bytree']= 0.7
param['min_child_weight'] = 0
param['alpha'] = 2
param['lambda'] = 2
param['booster'] = "gbtree"

watchlist  = [(dtrain, "train"),(dvalid, "valid")]
num_round = 10000
early_stopping_rounds=10
bst = xgb.train(param, dtrain, num_round, watchlist,early_stopping_rounds=early_stopping_rounds, verbose_eval=50)

dtest = xgb.DMatrix(X_test)
test_preds = pd.DataFrame(bst.predict(dtest))
Delivery_Time_Reverse_Lookup = ['10 minutes', '20 minutes', '30 minutes', '45 minutes', '65 minutes', '80 minutes', '120 minutes']
test_preds.columns = Delivery_Time_Reverse_Lookup
test_preds.head()

ID = data.groupby(['Restaurant'])['Delivery_Time'].agg(lambda x: pd.Series.mode(x)[0]).to_frame().reset_index()
submission = pd.DataFrame(test_preds.idxmax(axis=1), columns = ['Delivery_Time'])
submission['Restaurant'] = test['Restaurant']
submission = pd.merge(submission, ID, on='Restaurant', how='left')
submission['Delivery_Time'] = submission['Delivery_Time_y'].fillna(submission['Delivery_Time_x'])
submission = submission['Delivery_Time']
submission.to_excel(wd + 'submission.xlsx', header = True, index=False)

#LGBM
params = {
          "objective" : "multiclass",
          "num_class" : 7,
          "num_leaves" : 60,
          "max_depth": -1,
          "learning_rate" : 0.01,
          "bagging_fraction" : 0.9,  # subsample
          "feature_fraction" : 0.9,  # colsample_bytree
          "bagging_freq" : 5,        # subsample_freq
          "bagging_seed" : 2018,
          "verbosity" : -1 }


lgtrain = lgb.Dataset(X_train, y_train)
lgval = lgb.Dataset(X_val, y_val)
lgbmodel = lgb.train(params, lgtrain, 2000, valid_sets=[lgtrain, lgval], early_stopping_rounds=100, verbose_eval=20)

test_preds_lgbm = pd.DataFrame(lgbmodel.predict(X_test))
test_preds_lgbm.columns = Delivery_Time_Reverse_Lookup
test_preds_lgbm.head()

ID = data.groupby(['Restaurant'])['Delivery_Time'].agg(lambda x: pd.Series.mode(x)[0]).to_frame().reset_index()
submission = pd.DataFrame(test_preds_lgbm.idxmax(axis=1), columns = ['Delivery_Time'])
submission['Restaurant'] = test['Restaurant']
submission = pd.merge(submission, ID, on='Restaurant', how='left')
submission['Delivery_Time'] = submission['Delivery_Time_y'].fillna(submission['Delivery_Time_x'])
submission = submission['Delivery_Time']
submission.to_excel(wd + 'submission_LGBM.xlsx', header = True, index=False)


#submission = pd.DataFrame(test_preds.idxmax(axis=1), columns = ['Delivery_Time'])
#submission.to_excel(wd + 'submission.xlsx', header = True, index=False)


