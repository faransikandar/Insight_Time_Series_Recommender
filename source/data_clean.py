# -*- coding: utf-8 -*-
"""
Faran Sikandar
Insight AI.SV19B
06/08/2019

Project: Net_Align
Description: Using Representation Learning to Improve Recommender Systems for Economic Diversification

Data: Atlas of Economic Complexity

Notes:
- Recommender system code inspired from https://github.com/WillKoehrsen/wikipedia-data-science/blob/master/notebooks/Book%20Recommendation%20System.ipynb

Based off of original Colab file:

rec_sys_simple.ipynb

https://colab.research.google.com/drive/1P64VIbq6-FWVKYo503NT4y-_GgaIilgD

"""

"""# Imports and Setup

"""

#%%
# check relevant TF, keras, and GPU connections

# show which version of TF working
# !pip show tensorflow

# show which version of keras
# !pip show keras

'''
# check GPU connection
import tensorflow as tf
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))
'''

#%%
# import libraries
import os
import sys
import time
import warnings
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import re
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
from collections import Counter, defaultdict, OrderedDict
from itertools import chain
from keras import optimizers, regularizers
from keras.callbacks import ModelCheckpoint
from keras.layers import advanced_activations, Concatenate, Dense, Dot, Dropout, Embedding, Flatten, Input, LSTM, Reshape
from keras.models import load_model, Model, Sequential
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split

#%%
"""
## Setup Google Drive + Get the Data
"""

'''
# mount Google Drive locally
from google.colab import drive
drive.mount('/content/gdrive')
'''

#%%
from source.data_load import *

dict_data = data_load()
dict_data

hdf = dict_data['hdf']
#data_clean = dict_data['data_clean']

#%%
# preview keys in HDF
hdf.keys()

# testing

hdf.get('/country_hsproduct4digit_year').head()

# extract country summaries
df_country = hdf.get('/country')
print(df_country.shape)
df_country.head()

# extract country hs product lookbacks
df2_lookback = hdf.get('/country_hsproduct2digit_lookback')
df4_lookback = hdf.get('/country_hsproduct4digit_lookback') # there is no 6digit lookback
df4_lookback.head()

# extract country summaries lookback
df_country_lookback = hdf.get('/country_lookback')
df_country_lookback.tail()

# compare country + lookback shapes
print(df_country.shape)
print(df_country_lookback.shape)

# look at just one set of lookbacks for a given year
df_country_lookback3 = df_country_lookback[df_country_lookback['lookback_year'] == 2014]
print(df_country_lookback3.shape)

# extract classes - index is the product_id in other tables
df_classes = hdf.get('/classifications/hs_product')
type(df_classes)
print(df_classes.shape)
df2_classes = df_classes[df_classes['level'] == '2digit']
df4_classes = df_classes[df_classes['level'] == '4digit']
df6_classes = df_classes[df_classes['level'] == '6digit']

print(df6_classes.shape)
df6_classes.head()

# get locations - index is location_id in other tables
df_locations = hdf.get('/classifications/location')
print(df_locations.shape)
df_locations.head()

# extract hs product data
df2 = hdf.get('/country_hsproduct2digit_year')
df4 = hdf.get('/country_hsproduct4digit_year')
df6 = hdf.get('/country_hsproduct6digit_year')

print(df6.shape)
df6.head()
len(df6['product_id'].unique())

#%%
# clean the NaNs
try:
    df2 = df2.fillna(0)
except:
    df2

df2

df4 = df4.fillna(0)
df6 = df6.fillna(0)

# clean the negatives
mask = df2['export_value'] < 0
df2 = df2[~mask]

mask = df4['export_value'] < 0
df4 = df4[~mask]

mask = df6['export_value'] < 0
df6 = df6[~mask]

#%%
# Calculate totals by country and year

# df6.groupby(['location_id','year']).sum().reset_index() # if don't do reset_index(), then loc and year b/c part of index - does all columns

def group_sum(df, groups, targets, reset_index = True):
    '''
    input:  data = pandas df to groupby sum
            groups = list of features to groupbym e.g. ['location_id','year']
            targets = list variables to sum e.g. ['export_value']
    output: groupby sum
    '''
    if reset_index == True:
        df_groupsum = df.groupby(groups)[targets].sum().reset_index() # can also do .agg('sum')
    else:
        df_groupsum = df.groupby(groups)[targets].sum()
    return df_groupsum

'''
# MASTER FUNCTION FOR DATA CLEANING
def clean_data(df):
    df_groupsum = ( group_sum(df=df,groups=['location_id','year'],targets=['export_value','import_value'])
    .rename(index=str, columns={'export_value':'export_total_loc_year', 'import_value':'import_total_loc_year'}) )

    df_groupsum2 = ( 2 * group_sum(df=df,groups=['location_id','year'],targets=['export_value','import_value'])
    .rename(index=str, columns={'export_value':'export_total_loc_year', 'import_value':'import_total_loc_year'}) )

    return {'1':df_groupsum, '2':df_groupsum2}

clean_data(df6).keys()

df_group_x, df_group_x2 = clean_data(df=df6)

df_group_x
'''

# sum the exports/imports, by location and year - will be used for improved normalization by country
df6_groupsum = ( group_sum(df=df6,groups=['location_id','year'],targets=['export_value','import_value'])
.rename(index=str, columns={'export_value':'export_total_loc_year', 'import_value':'import_total_loc_year'}) )

df6_groupsum

df6_groupsum.describe()

#%%
def data_filter(df,filter,values):
    '''
    input:  df: pandas df to cut
            filter: single var to filter by e.g. year
            value: list or any iterable of filter value(s)
    output: df filtered by value
    '''
    df_filter = df.loc[df[filter].isin(values)]
    return df_filter

#%%
# filter the data for a 10 year TRAIN range
df6_95_04 = data_filter(df=df6,filter='year',values=range(1995,2005))
df6_95_04.head(10)

# filter the data for a 10 year TEST range
df6_05_14 = data_filter(df=df6,filter='year',values=range(2005,2015))
df6_05_14.head(10)

# TRAIN sum the exports/imports across the FIRST half of the time slice - for trend analysis
df6_95_04_sum1 = ( df6_95_04.loc[df6_95_04['year'].isin(range(1995,2000))].groupby(['location_id','product_id'])['export_value','import_value']
.sum().reset_index().rename(index=str, columns={'export_value':'export_period1', 'import_value':'import_period1'}) )

df6_95_04_sum1

# TRAIN sum the exports/imports across the SECOND half of the time slice - for trend analysis
df6_95_04_sum2 = ( df6_95_04.loc[df6_95_04['year'].isin(range(2000,2005))].groupby(['location_id','product_id'])['export_value','import_value']
.sum().reset_index().rename(index=str, columns={'export_value':'export_period2', 'import_value':'import_period2'}) )

df6_95_04_sum2

# TRAIN total exports/imports for a given product over ENTIRE 10 year period
df6_95_04_sum_total = ( df6_95_04.loc[df6_95_04['year'].isin(range(1995,2005))].groupby(['location_id','product_id'])['export_value','import_value']
.sum().reset_index().rename(index=str, columns={'export_value':'export_period_total', 'import_value':'import_period_total'}) )

df6_95_04_sum_total

# TEST sum the exports/imports across the FIRST half of the time slice - for trend analysis
df6_05_14_sum1 = ( df6_05_14.loc[df6_05_14['year'].isin(range(2005,2010))].groupby(['location_id','product_id'])['export_value','import_value']
.sum().reset_index().rename(index=str, columns={'export_value':'export_period1', 'import_value':'import_period1'}) )

df6_05_14_sum1

# TEST sum the exports/imports across the SECOND half of the time slice - for trend analysis
df6_05_14_sum2 = ( df6_05_14.loc[df6_05_14['year'].isin(range(2010,2015))].groupby(['location_id','product_id'])['export_value','import_value']
.sum().reset_index().rename(index=str, columns={'export_value':'export_period2', 'import_value':'import_period2'}) )

df6_05_14_sum2

# TEST total exports/imports for a given product over ENTIRE 10 year period
df6_05_14_sum_total = ( df6_05_14.loc[df6_05_14['year'].isin(range(2005,2015))].groupby(['location_id','product_id'])['export_value','import_value']
.sum().reset_index().rename(index=str, columns={'export_value':'export_period_total', 'import_value':'import_period_total'}) )

df6_05_14_sum_total

#%%
# calculate and merge sum2, period_total and export/import trends back into sum1 df; fill NaNs with 0 (if 0 base value)
df6_95_04_trend = ( df6_95_04_sum1.assign( export_period2 = df6_95_04_sum2['export_period2'], import_period2 = df6_95_04_sum2['import_period2'],
export_trend = lambda x: ((x.export_period2 - df6_95_04_sum1['export_period1'])/x.export_period1).fillna(0),
import_trend = lambda x: ((x.import_period2 - df6_95_04_sum1['import_period1'])/x.import_period1).fillna(0),
export_period_total = df6_95_04_sum_total['export_period_total'], import_period_total = df6_95_04_sum_total['import_period_total'] ) )

df6_05_14_trend = ( df6_05_14_sum1.assign( export_period2 = df6_05_14_sum2['export_period2'], import_period2 = df6_05_14_sum2['import_period2'],
export_trend = lambda x: ((x.export_period2 - df6_05_14_sum1['export_period1'])/x.export_period1).fillna(0),
import_trend = lambda x: ((x.import_period2 - df6_05_14_sum1['import_period1'])/x.import_period1).fillna(0),
export_period_total = df6_05_14_sum_total['export_period_total'], import_period_total = df6_05_14_sum_total['import_period_total'] ) )

# how to use assign to create multiple values in df
# df = df.assign(Val10_minus_Val1 = df['Val10'] - df['Val1'], log_result = lambda x: np.log(x.Val10_minus_Val1) )

df6_95_04_trend
df6_05_14_trend

#%%

# impute export inf/-inf with max/min trend for 95_04
mask_pos = df6_95_04_trend['export_trend'] != np.inf
mask_pos
mask_neg = df6_95_04_trend['export_trend'] != -np.inf
mask_neg
df6_95_04_trend[~mask_pos]

#df6_95_04_trend.loc[~mask_pos, 'export_trend'] = df6_95_04_trend.loc[mask_pos, 'export_trend'].max() # old method
#df6_95_04_trend.loc[~mask_neg, 'export_trend'] = df6_95_04_trend.loc[mask_neg, 'export_trend'].min()
df6_95_04_trend['export_trend'] = np.where(~mask_pos, df6_95_04_trend['export_period2'], df6_95_04_trend['export_trend']) # if div by 0, replaces inf w/ export_period2 value
df6_95_04_trend['export_trend'] = np.where(~mask_neg, -df6_95_04_trend['export_period2'], df6_95_04_trend['export_trend']) # if div by 0, replaces -inf w/ -export_period2 value

# impute export inf/-inf with max/min trend for 05_14
mask_pos = df6_05_14_trend['export_trend'] != np.inf
mask_pos
mask_neg = df6_05_14_trend['export_trend'] != -np.inf
mask_neg
df6_05_14_trend[~mask_pos]

# df6_05_14_trend.loc[~mask_pos, 'export_trend'] = df6_05_14_trend.loc[mask_pos, 'export_trend'].max()
# df6_05_14_trend.loc[~mask_neg, 'export_trend'] = df6_05_14_trend.loc[mask_neg, 'export_trend'].min()
df6_05_14_trend['export_trend'] = np.where(~mask_pos, df6_05_14_trend['export_period2'], df6_05_14_trend['export_trend']) # if div by 0, replaces inf w/ export_period2 value
df6_05_14_trend['export_trend'] = np.where(~mask_neg, -df6_05_14_trend['export_period2'], df6_05_14_trend['export_trend']) # if div by 0, replaces -inf w/ -export_period2 value

df6_05_14_trend[~mask_pos]

# impute import inf/-inf with max/min trend for 95_04
mask_pos = df6_95_04_trend['import_trend'] != np.inf
mask_pos
mask_neg = df6_95_04_trend['import_trend'] != -np.inf
mask_neg

df6_95_04_trend[~mask_neg]

# df6_95_04_trend.loc[~mask_pos, 'import_trend'] = df6_95_04_trend.loc[mask_pos, 'import_trend'].max()
# df6_95_04_trend.loc[~mask_neg, 'import_trend'] = df6_95_04_trend.loc[mask_neg, 'import_trend'].min()
df6_95_04_trend['import_trend'] = np.where(~mask_pos, df6_95_04_trend['import_period2'], df6_95_04_trend['import_trend']) # if div by 0, replaces inf w/ export_period2 value
df6_95_04_trend['import_trend'] = np.where(~mask_neg, -df6_95_04_trend['import_period2'], df6_95_04_trend['import_trend']) # if div by 0, replaces -inf w/ -export_period2 value

df6_95_04_trend[~mask_pos]

# impute import inf/-inf with max/min trend for 05_14
mask_pos = df6_05_14_trend['import_trend'] != np.inf
mask_pos
mask_neg = df6_05_14_trend['import_trend'] != -np.inf
mask_neg
df6_05_14_trend[~mask_neg]

# df6_05_14_trend.loc[~mask_pos, 'import_trend'] = df6_05_14_trend.loc[mask_pos, 'import_trend'].max()
# df6_05_14_trend.loc[~mask_neg, 'import_trend'] = df6_05_14_trend.loc[mask_neg, 'import_trend'].min()
df6_05_14_trend['import_trend'] = np.where(~mask_pos, df6_05_14_trend['import_period2'], df6_05_14_trend['import_trend']) # if div by 0, replaces inf w/ export_period2 value
df6_05_14_trend['import_trend'] = np.where(~mask_neg, -df6_05_14_trend['import_period2'], df6_05_14_trend['import_trend']) # if div by 0, replaces -inf w/ -export_period2 value


df6_95_04_trend.describe()
df6_95_04_trend

#%%
# merge df6_95_04_trend back into d56_95_04 by location and product (will be repeats of summed values)
train = pd.merge(df6_95_04, df6_groupsum, on=['location_id','year'], how='inner')
train = pd.merge(train, df6_95_04_trend, on=['location_id','product_id'], how='inner')
print(train.shape)
train

# merge df6_05_14_trend back into d56_05_14 by location and product (will be repeats of summed values)
test = pd.merge(df6_05_14, df6_groupsum, on=['location_id','year'], how='inner')
test = pd.merge(test, df6_05_14_trend, on=['location_id','product_id'], how='inner')
print(test.shape)
test

#%%
# Define train and test - make sure to normalize AFTER this so as not to have data leakage
cols = ['location_id','product_id','year','export_value','export_total_loc_year','export_period1','export_period2','export_period_total','export_trend']
train = train.copy(deep=True)
train = train[cols]
train

test = test[cols]
test = test.copy(deep=True)
test

#%%
# Calculate product percent of total exports for that country and year
train['export_pct'] = (train['export_value']/train['export_total_loc_year'])
train.head()

test['export_pct'] = (test['export_value']/test['export_total_loc_year'])
test.head()


#%%
# normalize by country and year - this may be redundant since we already made export_pct
def norm_minmax(data,targets):
    return (data[targets]-data[targets].min())/(data[targets].max()-data[targets].min())

def norm_std(data,targets):
    return (data[targets]-data[targets].mean())/(data[targets].std())

#%%
# norm across all countries and years
train['export_val_norm_all'] = norm_minmax(data=train,targets=['export_value'])
train['export_val_std_all'] = norm_std(data=train,targets=['export_value'])
train['export_pct_norm_all'] = norm_minmax(data=train,targets=['export_pct'])
train['export_pct_std_all'] = norm_std(data=train,targets=['export_pct'])
train.describe()
train

test['export_val_norm_all'] = norm_minmax(data=test,targets=['export_value'])
test['export_val_std_all'] = norm_std(data=test,targets=['export_value'])
test['export_pct_norm_all'] = norm_minmax(data=test,targets=['export_pct'])
test['export_pct_std_all'] = norm_std(data=test,targets=['export_pct'])
test.describe()
test


#%%
# normalize by country and year ??? doesn't seem to get me what I want - possible that you don't WANT to normalize by country and year, because perhaps overall global trade of goods is more important

train_val_norm = ( train.groupby(['location_id','year']).apply(norm_minmax, targets='export_value').to_frame()
.rename(index=str, columns={'export_value':'export_val_norm'}).reset_index() )

train_val_std = ( train.groupby(['location_id','year']).apply(norm_std, targets='export_value').to_frame()
.rename(index=str, columns={'export_value':'export_val_std'}).reset_index() )

train_pct_norm = ( train.groupby(['location_id','year']).apply(norm_minmax, targets='export_pct').to_frame()
.rename(index=str, columns={'export_pct':'export_pct_norm'}).reset_index() )

train_pct_std = ( train.groupby(['location_id','year']).apply(norm_std, targets='export_pct').to_frame()
.rename(index=str, columns={'export_pct':'export_pct_std'}).reset_index() )

train_trend_norm = ( train.groupby(['location_id']).apply(norm_minmax, targets='export_trend').to_frame()
.rename(index=str, columns={'export_trend':'export_trend_norm'}).reset_index() )

train_trend_std = ( train.groupby(['location_id']).apply(norm_std, targets='export_trend').to_frame()
.rename(index=str, columns={'export_trend':'export_trend_std'}).reset_index() )

test_val_norm = ( test.groupby(['location_id','year']).apply(norm_minmax, targets='export_value').to_frame()
.rename(index=str, columns={'export_value':'export_val_norm'}).reset_index() )

test_val_std = ( test.groupby(['location_id','year']).apply(norm_std, targets='export_value').to_frame()
.rename(index=str, columns={'export_value':'export_val_std'}).reset_index() )

test_pct_norm = ( test.groupby(['location_id','year']).apply(norm_minmax, targets='export_pct').to_frame()
.rename(index=str, columns={'export_pct':'export_pct_norm'}).reset_index() )

test_pct_std = ( test.groupby(['location_id','year']).apply(norm_std, targets='export_pct').to_frame()
.rename(index=str, columns={'export_pct':'export_pct_std'}).reset_index() )

test_trend_norm = ( test.groupby(['location_id']).apply(norm_minmax, targets='export_trend').to_frame()
.rename(index=str, columns={'export_trend':'export_trend_norm'}).reset_index() )

test_trend_std = ( test.groupby(['location_id']).apply(norm_std, targets='export_trend').to_frame()
.rename(index=str, columns={'export_trend':'export_trend_std'}).reset_index() )

'''
# same as
# df6_.groupby(['location_id','year']).apply( lambda x: (x['export_pct']-x['export_pct'].min())/(x['export_pct'].max()-x['export_pct'].min()) )
# do product_id as well - otherwise indices lost?
# df6_train.groupby(['location_id','year','product_id']).apply(norm_minmax, targets='export_pct').to_frame().reset_index()

# df6_train.groupby(['location_id','year'])( (df6['export_pct']-df6['export_pct'].min())/(df6_train['export_pct'].max()-df6_train['export_pct'].min()) )
# df2_2007_norm['export_value'] = (df2_2007['export_value']-df2_2007['export_value'].min())/(df2_2007['export_value'].max()-df2_2007['export_value'].min())
# df6.groupby(['location_id','year'])['export_value'].sum().reset_index()
'''

train_pct_std
test_trend_norm.describe()

#%%

# merge the pct and trend norms in
train_temp = train.join([train_val_norm['export_val_norm'], train_val_std['export_val_std'], train_pct_norm['export_pct_norm'], train_pct_std['export_pct_std'], train_trend_norm['export_trend_norm'], train_trend_std['export_trend_std']])
#train_temp = pd.merge(train, train_pct_norm['export_pct_norm'], train_pct_std['export_pct_std'], train_trend_norm['export_trend_norm'], left_index=True, right_index=True)
train = train_temp
train

test_temp = test.join([test_val_norm['export_val_norm'], test_val_std['export_val_std'], test_pct_norm['export_pct_norm'], test_pct_std['export_pct_std'], test_trend_norm['export_trend_norm'], test_trend_std['export_trend_std']])
#test_temp = pd.merge(test, test_pct_norm['export_pct_norm'], train_pct_std['export_pct_std'], test_trend_norm['export_trend_norm'], left_index=True, right_index=True)
test = test_temp
test


#df6_train['export_pct_norm'] = (df6_train['export_pct']-df6_train['export_pct'].min())/(df6_train['export_pct'].max()-df6_train['export_pct'].min())


#%%
# Classify -1/0/1 for export trend for TRAIN
mask_pos = train['export_trend'] > 0
mask_neg = train['export_trend'] < 0
mask_zero = train['export_trend'] == 0

train['export_trend_class'] = np.select([mask_pos,mask_neg], [1,-1], default=0)

# Make percentile rank for export trend for TRAIN
train['export_trend_pct_rank'] = train['export_trend'].rank(pct=True)

# Classify -1/0/1 for export trend for TEST
mask_pos = test['export_trend'] > 0
mask_neg = test['export_trend'] < 0
mask_zero = test['export_trend'] == 0

test['export_trend_class'] = np.select([mask_pos,mask_neg], [1,-1], default=0)

# Make percentile rank for export trend for TEST
test['export_trend_pct_rank'] = test['export_trend'].rank(pct=True)

#%%
# Final NaN cleaning
train[train.isnull()]
mask = train.isnull() == True
train[mask]
train = train.fillna(0)
train.isnull().values.any()

test = test.fillna(0)
test.isnull().values.any()

#%%
## Export data to HDF5 and pickle

# export to HDF5

clean = ( {'train':train, 'test':test, 'df_country':df_country, 'df4_lookback':df4_lookback, 'df_country_lookback':df_country_lookback,
'df6_classes':df6_classes, 'df_locations':df_locations} )

for key, value in clean.items():
    print(key)

# always make train the first item in the dict
directory = os.path.dirname(os.path.abspath('data_clean.h5'))
directory
clean_filename = os.path.join(directory,'data/processed/data_clean.h5')
clean_filename

if os.path.exists(clean_filename):
    os.remove(clean_filename)

for k, v in clean.items():
    try:
        if k == 'train':
            v.to_hdf('data/processed/data_clean.h5', key=k)
        else:
            v.to_hdf('data/processed/data_clean.h5', key=k)
    except NotImplementedError:
        if k == 'train':
            v.to_hdf('data/processed/data_clean.h5', key=k, format='t')
        else:
            v.to_hdf('data/processed/data_clean.h5', key=k, format='t')

#%%

prep = ( {'df6':df6, 'df6_groupsum':df6_groupsum, 'df6_95_04':df6_95_04, 'df6_05_14':df6_05_14, 'df6_95_04_sum1':df6_95_04_sum1,
'df6_95_04_sum2':df6_95_04_sum2, 'df6_95_04_trend':df6_95_04_trend, 'df6_05_14_sum1':df6_05_14_sum1, 'df6_05_14_sum2':df6_05_14_sum2,
'df6_05_14_trend':df6_05_14_trend, 'train_pct_norm':train_pct_norm, 'train_pct_std':train_pct_std, 'train_trend_std':train_trend_std,
'test_pct_norm':test_pct_norm, 'test_pct_std':test_pct_std, 'test_trend_std':test_trend_std} )

for k, v in prep.items():
    try:
        if k == 'df6':
            v.to_hdf('data/preprocessed/data_prep.h5', key=k, mode='w')
        else:
            v.to_hdf('data/preprocessed/data_prep.h5', key=k)
    except NotImplementedError:
        if k == 'train':
            v.to_hdf('data/preprocessed/data_prep.h5', key=k, mode='w', format='t')
        else:
            v.to_hdf('data/preprocessed/data_prep.h5', key=k, format='t')

#%%
# TESTING 123

train['export_val_std_all'].var()

# visualize the data
train.describe()

#%%
test.describe()
