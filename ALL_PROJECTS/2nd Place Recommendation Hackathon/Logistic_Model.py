# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 18:15:03 2020

@author: kbhandari
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import gc
import os
import pickle
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
from scipy.sparse import vstack, csr_matrix, save_npz, load_npz
from sklearn.model_selection import train_test_split

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

wd = "C:/My Desktop/Hackathon/"
os.chdir(wd)

data = pd.read_csv(wd + "DL_MDL_UNIV.csv")
test = pd.read_csv(wd + "DL_MDL_UNIV_SCORE.csv")
data.drop([col for col in data.columns if '_sequence_' in col], axis=1, inplace=True)
test.drop([col for col in test.columns if '_sequence_' in col], axis=1, inplace=True)
data.drop("customer_id", axis=1, inplace=True)
dependent_variables = [col for col in data.columns if 'Target' in col]

parameters = {
    'solver': ['lbfgs'],
    'C': [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01],
    'l1_ratio': [0.001, 0.01]
}

#learn, validation = train_test_split(data, test_size = 0.05, random_state = 0)
X_train = data[[col for col in data.columns if col not in dependent_variables]].values
#X_train = learn[[col for col in learn.columns if col not in dependent_variables]].values
#X_val = validation[[col for col in validation.columns if col not in dependent_variables]].values

min_max_scaler = preprocessing.MinMaxScaler()
X_train = min_max_scaler.fit_transform(X_train)
#X_val = min_max_scaler.transform(X_val)

X_test = test[[col for col in test.columns if 'customer_id' not in col]].values
X_test = min_max_scaler.transform(X_test)

#X_train_matrix = csr_matrix(X_train, dtype='float32')
#X_val_matrix = csr_matrix(X_val, dtype='float32')
#X_test_matrix = csr_matrix(X_test, dtype='float32')
X_train_matrix = X_train
X_test_matrix = X_test

test_preds = pd.DataFrame(test['customer_id'])
for col in dependent_variables:
    print(col)
    
    y_train = data[[col]].values
#    y_val = validation[[col]].values
    
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
mods.to_csv(wd + "predictions_KB_Bayesian_Belief_Net.csv", index=False)



sub = pd.read_csv('submission.csv')
dependent_variables = [x for x in sub.columns[1:]]

# Ensemble 1
rfc = pd.read_csv(wd + "FINAL/" + "predictions_RS_AUG_model1_8.csv")
mods = pd.read_csv(wd + "FINAL/" + "predictions_KB_Bayesian_Belief_Net.csv")
tfidf = pd.read_csv(wd + "FINAL/" + "predictions_KB.csv")
test_preds = pd.DataFrame(mods['customer_id']).reset_index(drop=True)
for col in dependent_variables:
    test_preds[col] = 0.4*rfc[col] + 0.2*mods[col] + 0.4*tfidf[col]
    
test_preds = test_preds[sub.columns]
test_preds = test_preds.sort_values(by="customer_id")    
test_preds.to_csv(wd + "GOD_IS_ULTIMATE.csv", index=False)


#### Ensemble 2
#rfc = pd.read_csv(wd + "FINAL/" + "predictions_RS_AUG_model1_8.csv")
#mods = pd.read_csv(wd + "FINAL/" + "predictions_RS_AUG_model1_35.csv")
#tfidf = pd.read_csv(wd + "FINAL/" + "predictions_KB.csv")
#test_preds = pd.DataFrame(mods['customer_id']).reset_index(drop=True)
#for col in dependent_variables:
#    test_preds[col] = 0.4*rfc[col] + 0.2*mods[col] + 0.4*tfidf[col]
#    
#test_preds = test_preds[sub.columns]
#test_preds = test_preds.sort_values(by="customer_id")    
#test_preds.to_csv(wd + "GOD_IS_ULTIMATE.csv", index=False)
##
##
### Ensemble 3
#model_1 = pd.read_csv(wd + "FINAL/" + "predictions_KB_2_RF_0.6514.csv")
#model_2 = pd.read_csv(wd +  "predictions_KB_DL3.csv")
##model_3 = pd.read_csv(wd + "predictions_KB_2.csv")
#test_preds = pd.DataFrame(model_1['customer_id']).reset_index(drop=True)
#for col in dependent_variables:
#    test_preds[col] = 0.6*model_1[col] + 0.4*model_2[col]
#    
#test_preds = test_preds[sub.columns]
#test_preds = test_preds.sort_values(by="customer_id")    
#test_preds.to_csv(wd + "E1.csv", index=False)

### Ensemble 4
#model_1 = pd.read_csv(wd + "predictions_KB_NEW_L1.csv")
#model_2 = pd.read_csv(wd + "predictions_KB_NEW_1.csv")
#test_preds = pd.DataFrame(model_1['customer_id']).reset_index(drop=True)
#for col in dependent_variables:
#    test_preds[col] = 0.35*model_1[col] + 0.65*model_2[col]
#    
#test_preds = test_preds[sub.columns]
#test_preds = test_preds.sort_values(by="customer_id")    
#test_preds.to_csv(wd + "E2.csv", index=False)

