# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 09:42:51 2020

@author: kbhandari
"""

import pandas as pd
import numpy as np
from datetime import timedelta
import os

RunType = 'Score'

wd = "C:/My Desktop/Hackathon/"
os.chdir(wd)

raw_data=pd.read_csv('train.csv')
raw_data.head()

#raw_data.isna().sum()

raw_data['InvoiceDate']=pd.to_datetime(raw_data['InvoiceDate'])

raw_data.dropna(inplace=True)

raw_data.head(20)

#len(raw_data),len(raw_data['customer_id'])

## Deep Learning Features
ref_date = raw_data['InvoiceDate'].max()-timedelta(days=60)
dl_raw_data = raw_data[raw_data['InvoiceDate']<=ref_date]

dl_raw_data[['categorical_sequence_items','continuous_sequence_qty','continuous_sequence_amt']] = dl_raw_data[['item_id','item_qty','item_amt']].astype(str)

dl_raw_data['continuous_sequence_Days_Between_TXN'] = pd.to_datetime(dl_raw_data['InvoiceDate']).sub(pd.to_datetime(ref_date)).dt.days.astype(str)
agg_df = dl_raw_data.sort_values('InvoiceDate', ascending=False).groupby(['customer_id']).agg({
        'categorical_sequence_items': lambda s: ', '.join(s),
        'continuous_sequence_qty': lambda s: ', '.join(s),
        'continuous_sequence_amt': lambda s: ', '.join(s),
        'continuous_sequence_Days_Between_TXN': lambda s: ', '.join(s)
        }).reset_index()

sub=pd.read_csv('submission.csv')

target=[x[7:] for x in sub.columns[1:]]

#data = raw_data.copy()
data=raw_data[raw_data['item_id'].isin([x[7:] for x in sub.columns[1:]])]

if RunType.upper() == 'MODEL':
    rd=raw_data['InvoiceDate'].max()-timedelta(days=60)
else:
    rd=raw_data['InvoiceDate'].max()
rd
rd1=rd-timedelta(days=1)
rd7=rd-timedelta(days=7)
rd30=rd-timedelta(days=30)
rd60=rd-timedelta(days=60)
rd90=rd-timedelta(days=90)
rd360=rd-timedelta(days=360)

rd7,rd30,rd60,rd90,rd360

if RunType.upper() == 'MODEL':
    data=data[data['InvoiceDate']<=rd]

len(data)
data.reset_index(inplace=True,drop=['index'])
print(data.head())

def freq(dataset,reference_date,f,g,l):
    return dataset[dataset['InvoiceDate']>=reference_date].groupby(['customer_id','item_id'],as_index=False).agg({'transaction_id':'count','item_amt':'sum','item_qty':'sum'}).rename(columns={'transaction_id':f,'item_amt':g,'item_qty':l})
    
f1=freq(data,rd1,'F1','M1','Q1')
f7=freq(data,rd7,'F7','M7','Q7')
f30=freq(data,rd30,'F30','M30','Q30')
f60=freq(data,rd60,'F60','M60','Q60')
f90=freq(data,rd90,'F90','M90','Q90')
f360=freq(data,rd360,'F360','M360','Q360')
#f720=freq(data,rd720,'F720','M720','Q720')

# Last 3 transaction average, stdev
data['PrevInvoiceDate'] = data.sort_values(['InvoiceDate'],ascending=True).groupby(['customer_id','item_id'])['InvoiceDate'].shift(1)
data['T2InvoiceDate'] = data.sort_values(['InvoiceDate'],ascending=True).groupby(['customer_id','item_id'])['InvoiceDate'].shift(2)
data['T3InvoiceDate'] = data.sort_values(['InvoiceDate'],ascending=True).groupby(['customer_id','item_id'])['InvoiceDate'].shift(3)

data['DayDiff'] = (data['InvoiceDate'] - data['PrevInvoiceDate']).dt.days
data['DayDiff2'] = (data['InvoiceDate'] - data['T2InvoiceDate']).dt.days
data['DayDiff3'] = (data['InvoiceDate'] - data['T3InvoiceDate']).dt.days

# penalizing null values with 999
data['DayDiff'].fillna(999, inplace=True)
data['DayDiff2'].fillna(999, inplace=True)
data['DayDiff3'].fillna(999, inplace=True)

tx_day_diff = data.groupby(['customer_id','item_id']).agg({'DayDiff': ['mean', 'std'], 'DayDiff2': ['mean', 'std'], 'DayDiff3': ['mean', 'std']}).reset_index()
tx_day_diff.columns = ['customer_id', 'item_id', 'DayDiffMean', 'DayDiffStd', 'DayDiff2Mean', 'DayDiff2Std', 'DayDiff3Mean', 'DayDiff3Std']
tx_day_diff.fillna(999, inplace=True)

f=pd.merge(pd.merge(pd.merge(pd.merge(pd.merge(f1,f7,how='outer',on=['customer_id','item_id']),f30,how='outer',on=['customer_id','item_id']),f60,how='outer',on=['customer_id','item_id']),f90,how='outer',on=['customer_id','item_id']),f360,how='outer',on=['customer_id','item_id'])
f = pd.merge(f, tx_day_diff, how='outer', on=['customer_id','item_id'])
cids = pd.DataFrame(raw_data.customer_id.unique(), columns=['customer_id'])
len(cids.customer_id.unique())
len(data.customer_id.unique())
len(f.customer_id.unique())

#data[data['customer_id']==12348].sort_values('InvoiceDate',ascending=False)

#f[f['customer_id']==12348]

rd,rd7,rd30,rd60,rd90,rd360

c_r_l=data.groupby(['customer_id','item_id'],as_index=False).agg({'InvoiceDate':[lambda x: (rd-x.max()).days]})
c_r_f=data.groupby(['customer_id','item_id'],as_index=False).agg({'InvoiceDate':[lambda x: (rd-x.min()).days]})

#c_r_f.columns.map('_'.join).str.strip('_')

c_r_l.columns=c_r_l.columns.map('_'.join).str.strip('_')
c_r_f.columns=c_r_f.columns.map('_'.join).str.strip('_')

c_r_l.rename(columns={'InvoiceDate_<lambda>':'R_L'},inplace=True)
c_r_f.rename(columns={'InvoiceDate_<lambda>':'R_F'},inplace=True)

c_T_L=data.groupby(['customer_id'],as_index=False).agg({'InvoiceDate':[lambda x: (rd-x.max()).days]})
c_T_F=data.groupby(['customer_id'],as_index=False).agg({'InvoiceDate':[lambda x: (rd-x.min()).days]})

c_T_F

#c_T_L.columns.map('_'.join).str.strip('_')

c_T_L.columns=c_T_L.columns.map('_'.join).str.strip('_')
c_T_F.columns=c_T_F.columns.map('_'.join).str.strip('_')

c_T_L.rename(columns={'InvoiceDate_<lambda>':'T_L'},inplace=True)
c_T_F.rename(columns={'InvoiceDate_<lambda>':'T_F'},inplace=True)

f.head()

c_r_l.head()

c_r_f.head()

c_T_F.head()

c_T_L.head()

d=pd.merge(pd.merge(f,c_r_f,how='left',on=['customer_id','item_id']),c_r_l,how='left',on=['customer_id','item_id'])

d.head()

d=d.set_index(['customer_id','item_id']).unstack()

d.columns

d.columns=d.columns.map('_'.join).str.strip('_')

d.head()

d.reset_index(level=['customer_id'],inplace=True)

data_processed=pd.merge(pd.merge(d,c_T_F,how='left',on='customer_id'),c_T_L,how='left',on=['customer_id'])

data_processed=pd.merge(pd.DataFrame(raw_data['customer_id'].unique(), columns=['customer_id']),data_processed,how='left',on=['customer_id'])
data_processed.fillna(999 ,inplace=True)

# DV Creation
if RunType.upper() == 'MODEL':
    data1=raw_data[raw_data['InvoiceDate']>rd]
    
    data1.loc[:,'Target']=1
    
    data1.head()
    
    data1=data1[data1['item_id'].isin(target)]
    
    dv=data1.groupby(['customer_id','item_id']).agg({'Target':'count'}).unstack()
    
    for i in dv.columns:
        dv[i]=dv[i].apply(lambda x: 1 if x>0 else 0)
        
    dv
    
    dv.columns=dv.columns.map('_'.join)
    
    dv.reset_index(level=['customer_id'],inplace=True)

# Adding DL Features
data_processed = pd.merge(data_processed, agg_df, how='outer', on=['customer_id'])
data_processed.fillna("", inplace=True)

if RunType.upper() == 'MODEL':
    data_processed = pd.merge(data_processed, dv, how='inner',on=['customer_id'])
    data_processed.fillna(0, inplace=True)

#dv1=pd.merge(data_processed['customer_id'],dv,how='left',on=['customer_id'])
#
#dv1.fillna(0,inplace=True)
#
#dv1

data_processed.head()
if RunType.upper() == 'MODEL':
    data_processed.to_csv("DL_MDL_UNIV.csv", header=True, index=False)
else:
    model_univ = pd.read_csv(wd + "DL_MDL_UNIV.csv")
    cols_to_keep = model_univ.columns
    data_processed = data_processed[cols_to_keep]
    data_processed=data_processed[data_processed['customer_id'].isin(sub.customer_id)]
    data_processed.to_csv("DL_MDL_UNIV_SCORE.csv", header=True, index=False)