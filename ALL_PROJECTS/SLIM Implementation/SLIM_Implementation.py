# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 11:29:37 2020

@author: kbhandari
"""

import os
wd = "C:/Users/kbhandari/OneDrive - Epsilon/Desktop/Guitar Center Raw Data/ARRAYS/"
os.chdir(wd)

import numpy as np
import scipy.sparse as sps
from Base.Recommender_utils import check_matrix
from sklearn.linear_model import ElasticNet

from Base.BaseSimilarityMatrixRecommender import BaseSimilarityMatrixRecommender
import time, sys
import pandas as pd
from scipy.sparse import csr_matrix
import warnings

class SLIM(object):
    ##
    ## Initializations
    ##
    def __init__(self,parameters):
        self.RECOMMENDER_NAME = 'SLIMElasticNet'
        if 'data_path' in parameters:
            self.data_path = parameters['data_path']
            self.URM_train = np.zeros((100,100,0),np.int32)
        else:
            self.data_path=""
            print("Gimme data..")

    def fit(self, l1_ratio=0.1, alpha = 1.0, positive_only=True, topK = 100,
            verbose = True):

        assert l1_ratio>= 0 and l1_ratio<=1, "{}: l1_ratio must be between 0 and 1, provided value was {}".format(self.RECOMMENDER_NAME, l1_ratio)

        self.l1_ratio = l1_ratio
        self.positive_only = positive_only
        self.topK = topK

        
        # initialize the ElasticNet model
        self.model = ElasticNet(alpha=alpha,
                                l1_ratio=self.l1_ratio,
                                positive=self.positive_only,
                                fit_intercept=False,
                                copy_X=False,
                                precompute=True,
                                selection='random',
                                max_iter=100,
                                tol=1e-4)


        URM_train = check_matrix(self.URM_train, 'csc', dtype=np.float32)

        n_items = URM_train.shape[1]


        # Use array as it reduces memory requirements compared to lists
        dataBlock = 10000000

        rows = np.zeros(dataBlock, dtype=np.int32)
        cols = np.zeros(dataBlock, dtype=np.int32)
        values = np.zeros(dataBlock, dtype=np.float32)

        numCells = 0


        start_time = time.time()
        start_time_printBatch = start_time
        
        didNotConvergeList = []

        # fit each item's factors sequentially (not in parallel)
        for currentItem in range(n_items):

            # get the target column
            y = URM_train[:, currentItem].toarray()

            # set the j-th column of X to zero
            start_pos = URM_train.indptr[currentItem]
            end_pos = URM_train.indptr[currentItem + 1]

            current_item_data_backup = URM_train.data[start_pos: end_pos].copy()
            URM_train.data[start_pos: end_pos] = 0.0



            # fit one ElasticNet model per column
            with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
                warnings.simplefilter("always")
                self.model.fit(URM_train, y)
                    # Verify some things
                if len(w) == 1:
                    if "ConvergenceWarning" in str(w[-1].message):
                        didNotConvergeList.append(currentItem)
    
            # self.model.coef_ contains the coefficient of the ElasticNet model
            # let's keep only the non-zero values

            # Select topK values
            # Sorting is done in three steps. Faster then plain np.argsort for higher number of items
            # - Partition the data to extract the set of relevant items
            # - Sort only the relevant items
            # - Get the original item index

            # nonzero_model_coef_index = self.model.coef_.nonzero()[0]
            # nonzero_model_coef_value = self.model.coef_[nonzero_model_coef_index]

            nonzero_model_coef_index = self.model.sparse_coef_.indices
            nonzero_model_coef_value = self.model.sparse_coef_.data


            local_topK = min(len(nonzero_model_coef_value)-1, self.topK)

            relevant_items_partition = (-nonzero_model_coef_value).argpartition(local_topK)[0:local_topK]
            relevant_items_partition_sorting = np.argsort(-nonzero_model_coef_value[relevant_items_partition])
            ranking = relevant_items_partition[relevant_items_partition_sorting]


            for index in range(len(ranking)):

                if numCells == len(rows):
                    rows = np.concatenate((rows, np.zeros(dataBlock, dtype=np.int32)))
                    cols = np.concatenate((cols, np.zeros(dataBlock, dtype=np.int32)))
                    values = np.concatenate((values, np.zeros(dataBlock, dtype=np.float32)))


                rows[numCells] = nonzero_model_coef_index[ranking[index]]
                cols[numCells] = currentItem
                values[numCells] = nonzero_model_coef_value[ranking[index]]

                numCells += 1


            # finally, replace the original values of the j-th column
            URM_train.data[start_pos:end_pos] = current_item_data_backup


            if verbose and (time.time() - start_time_printBatch > 300 or currentItem == n_items-1):
                print("{}: Processed {} ( {:.2f}% ) in {:.2f} minutes. Items per second: {:.0f}".format(
                    self.RECOMMENDER_NAME,
                    currentItem+1,
                    100.0* float(currentItem+1)/n_items,
                    (time.time()-start_time)/60,
                    float(currentItem)/(time.time()-start_time)))

                sys.stdout.flush()
                sys.stderr.flush()

                start_time_printBatch = time.time()


        # generate the sparse weight matrix
        self.W_sparse = sps.csr_matrix((values[:numCells], (rows[:numCells], cols[:numCells])),
                                       shape=(n_items, n_items), dtype=np.float32)
        self.didNotConvergeList = didNotConvergeList
        
        
#cross_sell_data = pd.read_csv(wd + 'customer_product_item_count.csv') 
##Change column names
#cross_sell_data.columns = ['CID', 'MERCH_ENTITY_CD', 'num_ITEMS']
#cross_sell_data = cross_sell_data[cross_sell_data['CID']!=0]
#cross_sell_data.head()
        
cross_sell_data = pd.read_csv(wd + 'Scoring_Data/' + 'NUMBER_ITEMS_FROM_CATEGORY.csv')
cross_sell_data['num_ITEMS']=cross_sell_data.iloc[:,2:].sum(axis=1) 
cross_sell_data = cross_sell_data[['SHIPTO_ACCT_SRC_NBR', 'CATEGORY3_CD', 'num_ITEMS']]
#Change column names
cross_sell_data.columns = ['CID', 'MERCH_ENTITY_CD', 'num_ITEMS']
cross_sell_data = cross_sell_data[cross_sell_data['CID']!=0]
cross_sell_data.head()

#Encode Categories
item_index = cross_sell_data['MERCH_ENTITY_CD'].value_counts().reset_index(drop=False)
item_index.reset_index(drop=False, inplace=True)
item_index.columns = ['Index','MERCH_ENTITY_CD', 'Count']
item_index.drop('Count', axis=1, inplace=True)
item_index['Index'] = item_index['Index'] + 1
item_index.index =  item_index['MERCH_ENTITY_CD']
item_index.drop('MERCH_ENTITY_CD', axis=1, inplace=True)
reverse_item_index = item_index.reset_index(drop=False)
reverse_item_index.index = reverse_item_index['Index']
reverse_item_index.drop('Index', axis=1, inplace=True)
cross_sell_data['MERCH_ENTITY_CD'] = item_index.loc[cross_sell_data['MERCH_ENTITY_CD']].values
cross_sell_data.head()

unique_CID = pd.DataFrame(cross_sell_data['CID'].unique())
unique_CID['MERCH_ENTITY_CD'] = 9999 #Intercept Term
unique_CID['num_ITEMS'] = 1
unique_CID.columns
unique_CID = unique_CID.rename(columns={0: "CID"})
unique_CID.columns
cross_sell_data = cross_sell_data.append(unique_CID,sort=True)

cross_sell_data['count'] = cross_sell_data.groupby('CID')['CID'].transform('count')
cross_sell_data_gt5 = cross_sell_data[cross_sell_data['count']>15]
CID = np.array(cross_sell_data_gt5['CID'])
MERCH_ENTITY_CD = np.array(cross_sell_data_gt5['MERCH_ENTITY_CD'])
num_ITEMS = np.array(cross_sell_data_gt5['num_ITEMS'])

CID.shape, MERCH_ENTITY_CD.shape, num_ITEMS.shape
num_ITEMS = np.log(num_ITEMS+0.1)
num_ITEMS

#Create dictionery to convert CID to a user id for embedding that is indexed from 0 and is contigous
ids = np.unique(CID)
numCID = len(np.unique(CID))

id2CID = {id:i for i,id in zip(range(numCID),ids)}
a_id = np.zeros(CID.shape,dtype=np.int32)

for i in range(CID.shape[0]):
    a_id[i,] = id2CID[CID[i,]]

#Create dictionery to convert MERCH_ENTITY_CD to a id for embedding that is indexed from 0 and is contigous
ids = np.unique(np.array(cross_sell_data_gt5['MERCH_ENTITY_CD']))
numMERCH_ENTITY_CD = len(np.unique(np.array(cross_sell_data_gt5['MERCH_ENTITY_CD'])))

id2numMERCH_ENTITY_CD = {id:i for i,id in zip(range(numMERCH_ENTITY_CD),ids)}
b_id = np.zeros(MERCH_ENTITY_CD.shape,dtype=np.int32)

for i in range(MERCH_ENTITY_CD.shape[0]):
    b_id[i,] = id2numMERCH_ENTITY_CD[MERCH_ENTITY_CD[i,]]

a_id.shape,b_id.shape,numMERCH_ENTITY_CD

data = csr_matrix((num_ITEMS, (a_id, b_id)), shape=(numCID,numMERCH_ENTITY_CD))

model = SLIM({"data_path":"abc"})
model.URM_train = data
print(model.URM_train.shape)

model.fit(l1_ratio=0.1, alpha = 1.0, positive_only=True, topK = model.URM_train.shape[1])
print(model.didNotConvergeList)

import gc
gc.collect()

def writeCSChunk(model,cross_sell_data,chunk_id):
    cross_sell_data_slice = cross_sell_data[cross_sell_data['chunk']==chunk_id]
    
    CID = np.array(cross_sell_data_slice['CID'])
    MERCH_ENTITY_CD = np.array(cross_sell_data_slice['MERCH_ENTITY_CD'])
    num_ITEMS = np.array(cross_sell_data_slice['num_ITEMS'])
    num_ITEMS = np.log(num_ITEMS+0.1)

    #Create dictionery to convert CID to a user id for embedding that is indexed from 0 and is contigous
    ids = np.unique(CID)
    numCID = len(np.unique(CID))

    id2CID = {id:i for i,id in zip(range(numCID),ids)}
    a_id = np.zeros(CID.shape,dtype=np.int32)

    for i in range(CID.shape[0]):
        a_id[i,] = id2CID[CID[i,]]

    b_id = np.zeros(MERCH_ENTITY_CD.shape,dtype=np.int32)

    for i in range(MERCH_ENTITY_CD.shape[0]):
        b_id[i,] = id2numMERCH_ENTITY_CD[MERCH_ENTITY_CD[i,]]

    data = csr_matrix((num_ITEMS, (a_id, b_id)), shape=(numCID,numMERCH_ENTITY_CD))
    user_id_array = data[:,:]
    item_scores = user_id_array.dot(model.W_sparse).toarray()
    df=pd.DataFrame(data=item_scores[0:,0:],index=[i for i in range(item_scores.shape[0])],columns=['item'+str(i) for i in range(item_scores.shape[1])])
    df['CID'] = np.unique(CID)
    l = pd.wide_to_long(df, stubnames='item', i=['CID'],j='MERCH_ENTITY_CD')
    l=l.reset_index()
    #
    # Need to get actual MERCH_ENTITY_CD back using dictionery rather than assuming a particular order
    #
    l = l[l['MERCH_ENTITY_CD']!=len(np.unique(np.array(cross_sell_data_gt5['MERCH_ENTITY_CD'])))-1]
    l['MERCH_ENTITY_CD'] = l['MERCH_ENTITY_CD'] + 1
    l=l.rename(columns={"item": "score"})    
#    l = l.merge(cross_sell_data_slice, on=['CID','MERCH_ENTITY_CD'], how='left')
#    l = l[l['num_ITEMS'].isnull()]
#    l = l.drop(columns=['num_ITEMS','chunk'])
    
    #Keshav Add
    #Remove 0 scores
    l = l[l['score']!=0]
    # Replace MERCH_ENTITY_CD back to original products
    m = reverse_item_index.to_dict()['MERCH_ENTITY_CD']
    l = l.replace({"MERCH_ENTITY_CD": m})
    
    l.to_csv(wd + 'scores_slim_'+str(chunk_id)+'.csv',index=False)
    del(l)
    gc.collect()
    

from tqdm import tqdm
from random import randint

numChunks = 1
CIDs = pd.DataFrame(cross_sell_data['CID'].unique())
CIDs = CIDs.rename(columns={0: "CID"})
CIDs = CIDs.sample(frac=1)
CIDs['random'] = np.arange(len(CIDs))
CIDs['chunk'] = 0
CIDs['random'] = CIDs['random']/len(CIDs.index)

interval=int(100/numChunks)/100
remainder=(100%numChunks)/100
for index in range(numChunks):
    CIDs.loc[(CIDs['random']>index*interval) & (CIDs['random']<=(index*interval+ interval)),'chunk'] = index
if remainder>0:
    CIDs.loc[(CIDs['random']>numChunks*interval) & (CIDs['random']<=(numChunks*interval + remainder)),'chunk'] = numChunks
    
cross_sell_data = cross_sell_data.merge(CIDs, on='CID', how='left')
unique_MERCH_ENTITY_CD = pd.DataFrame(cross_sell_data_gt5['MERCH_ENTITY_CD'].unique())
unique_MERCH_ENTITY_CD = unique_MERCH_ENTITY_CD.rename(columns={0: "MERCH_ENTITY_CD"})
unique_MERCH_ENTITY_CD['PRODUCT_IN_SCOPE']=1
cross_sell_data = cross_sell_data.merge(unique_MERCH_ENTITY_CD, on='MERCH_ENTITY_CD', how='left')
cross_sell_data = cross_sell_data.dropna()
cross_sell_data = cross_sell_data.drop(columns=['random','count','PRODUCT_IN_SCOPE'])

for i in tqdm(range(numChunks)):
    writeCSChunk(model,cross_sell_data,i)
    
#writeCSChunk(model,cross_sell_data,5)


# Scoring
test = pd.read_csv(wd + 'Scoring_Data/' + 'NUMBER_ITEMS_FROM_CATEGORY.csv')
test['num_ITEMS']=test.iloc[:,2:].sum(axis=1) 
test = test[['SHIPTO_ACCT_SRC_NBR', 'CATEGORY3_CD', 'num_ITEMS']]
#Change column names
test.columns = ['CID', 'MERCH_ENTITY_CD', 'num_ITEMS']
test = test[test['CID']!=0]
test.head()
#Change category names
n = item_index.to_dict()['Index']
test = test.replace({"MERCH_ENTITY_CD": n})
test.head()

unique_CID = pd.DataFrame(test['CID'].unique())
unique_CID['MERCH_ENTITY_CD'] = 9999 #Intercept Term
unique_CID['num_ITEMS'] = 1
unique_CID.columns
unique_CID = unique_CID.rename(columns={0: "CID"})
unique_CID.columns
test = test.append(unique_CID,sort=True)

numChunks = 1
CIDs = pd.DataFrame(test['CID'].unique())
CIDs = CIDs.rename(columns={0: "CID"})
CIDs = CIDs.sample(frac=1)
CIDs['random'] = np.arange(len(CIDs))
CIDs['chunk'] = 0
CIDs['random'] = CIDs['random']/len(CIDs.index)

interval=int(100/numChunks)/100
remainder=(100%numChunks)/100
for index in range(numChunks):
    CIDs.loc[(CIDs['random']>index*interval) & (CIDs['random']<=(index*interval+ interval)),'chunk'] = index
if remainder>0:
    CIDs.loc[(CIDs['random']>numChunks*interval) & (CIDs['random']<=(numChunks*interval + remainder)),'chunk'] = numChunks

test = test.merge(CIDs, on='CID', how='left')
unique_MERCH_ENTITY_CD = pd.DataFrame(cross_sell_data_gt5['MERCH_ENTITY_CD'].unique())
unique_MERCH_ENTITY_CD = unique_MERCH_ENTITY_CD.rename(columns={0: "MERCH_ENTITY_CD"})
unique_MERCH_ENTITY_CD['PRODUCT_IN_SCOPE']=1
test = test.merge(unique_MERCH_ENTITY_CD, on='MERCH_ENTITY_CD', how='left')
test = test.dropna()
test = test.drop(columns=['random','PRODUCT_IN_SCOPE'])

for i in tqdm(range(numChunks)):
    writeCSChunk(model,test,i)





