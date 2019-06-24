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
import nltk
import numpy as np
import pandas as pd
import pickle
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
from collections import Counter, defaultdict, OrderedDict
from itertools import chain
from nltk.tokenize import sent_tokenize, word_tokenize
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
def get_data():

    # load paths
    def load_paths(gdrive = False, ide = False):
        '''
        input: choose whether you are running the script in gdrive, as a shell, or locally (e.g. ide)
        output: relative paths for directories, hdf_filename || can add other directories and files as needed
        '''
        cwd = os.getcwd()
        cwd
        # for shell
        try:
            directory = os.path.dirname(os.path.abspath(__file__))
            hdf_filename = os.path.join(directory,'../data/raw/data_17_0.h5')
            directory = os.path.dirname(os.path.abspath(__file__))
            clean_filename = os.path.join(directory,'../data/processed/data_clean.h5')
        except NameError:
            # for gdrive
            if gdrive == True:
                directory = os.path.dirname(os.path.abspath('data_17_0.h5'))
                hdf_filename = '/content/gdrive/My Drive/Colab Notebooks/Insight_Net_Align/data/raw/data_17_0.h5'
                clean_filename = '/content/gdrive/My Drive/Colab Notebooks/Insight_Net_Align/data/processed/data_clean.h5'
            # for local (e.g. IDE)
            if ide == True:
                directory = os.path.dirname(os.path.abspath('data_17_0.h5'))
                directory
                hdf_filename = os.path.join(directory,'data/raw/data_17_0.h5')
                hdf_filename
                clean_filename = os.path.join(directory,'data/processed/data_clean.h5')
                clean_filename
        dict_paths_def = {'cwd':cwd, 'directory':directory, 'hdf_filename':hdf_filename, 'clean_filename':clean_filename}
        return dict_paths_def

    dict_paths = load_paths(gdrive = False, ide = True)

    # check path names
    dict_paths

    # load the data
    def load_data():
        hdf = pd.HDFStore(dict_paths['hdf_filename'], mode='r')
        data_clean = pd.HDFStore(dict_paths['clean_filename'], mode='r')
        dict_load = {'hdf':hdf,'data_clean':data_clean}
        return dict_load

    hdf = load_data()['hdf']
    data_clean = load_data()['data_clean']

    #%%
    data_clean.keys()

    #%%
    #df6_classes = data_clean.get('/df6_classes')
    #df_locations = data_clean.get('/df_locations')
    test = data_clean.get('/test')
    train = data_clean.get('/train')

    data_get = {'train':train, 'test':test}

    data_clean.close()

    return data_get

get_data()

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
df6 = df2[~mask]

mask = df4['export_value'] < 0
df6 = df4[~mask]

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

# MASTER FUNCTION FOR DATA CLEANING
def clean_data(df):
    df_groupsum = ( group_sum(df=df,groups=['location_id','year'],targets=['export_value','import_value'])
    .rename(index=str, columns={'export_value':'export_total', 'import_value':'import_total'}) )

    df_groupsum2 = ( 2 * group_sum(df=df,groups=['location_id','year'],targets=['export_value','import_value'])
    .rename(index=str, columns={'export_value':'export_total', 'import_value':'import_total'}) )

    return {'1':df_groupsum, '2':df_groupsum2}

clean_data(df6).keys()

df_group_x, df_group_x2 = clean_data(df=df6)

df_group_x

# sum the exports/imports, by location and year - will be used for improved normalization by country
df6_groupsum = ( group_sum(df=df6,groups=['location_id','year'],targets=['export_value','import_value'])
.rename(index=str, columns={'export_value':'export_total', 'import_value':'import_total'}) )

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

# TEST sum the exports/imports across the FIRST half of the time slice - for trend analysis
df6_05_14_sum1 = ( df6_05_14.loc[df6_05_14['year'].isin(range(2005,2010))].groupby(['location_id','product_id'])['export_value','import_value']
.sum().reset_index().rename(index=str, columns={'export_value':'export_period1', 'import_value':'import_period1'}) )

df6_05_14_sum1

# TEST sum the exports/imports across the SECOND half of the time slice - for trend analysis
df6_05_14_sum2 = ( df6_05_14.loc[df6_05_14['year'].isin(range(2010,2015))].groupby(['location_id','product_id'])['export_value','import_value']
.sum().reset_index().rename(index=str, columns={'export_value':'export_period2', 'import_value':'import_period2'}) )

df6_05_14_sum2

#%%
# calculate and merge sum2 and export/import trends back into sum1 df; fill NaNs with 0 (if 0 base value)
df6_95_04_trend = ( df6_95_04_sum1.assign( export_period2 = df6_95_04_sum2['export_period2'], import_period2 = df6_95_04_sum2['import_period2'],
export_trend = lambda x: ((x.export_period2 - df6_95_04_sum1['export_period1'])/x.export_period2).fillna(0),
import_trend = lambda x: ((x.import_period2 - df6_95_04_sum1['import_period1'])/x.import_period2).fillna(0) ) )

df6_05_14_trend = ( df6_05_14_sum1.assign( export_period2 = df6_05_14_sum2['export_period2'], import_period2 = df6_05_14_sum2['import_period2'],
export_trend = lambda x: ((x.export_period2 - df6_05_14_sum1['export_period1'])/x.export_period2).fillna(0),
import_trend = lambda x: ((x.import_period2 - df6_05_14_sum1['import_period1'])/x.import_period2).fillna(0) ) )

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
df6_95_04_trend[~mask_neg]

df6_95_04_trend.loc[~mask_pos, 'export_trend'] = df6_95_04_trend.loc[mask_pos, 'export_trend'].max()
df6_95_04_trend.loc[~mask_neg, 'export_trend'] = df6_95_04_trend.loc[mask_neg, 'export_trend'].min()

# impute export inf/-inf with max/min trend for 05_14
mask_pos = df6_05_14_trend['export_trend'] != np.inf
mask_pos
mask_neg = df6_05_14_trend['export_trend'] != -np.inf
mask_neg
df6_05_14_trend[~mask_neg]

df6_05_14_trend.loc[~mask_pos, 'export_trend'] = df6_05_14_trend.loc[mask_pos, 'export_trend'].max()
df6_05_14_trend.loc[~mask_neg, 'export_trend'] = df6_05_14_trend.loc[mask_neg, 'export_trend'].min()

# impute import inf/-inf with max/min trend for 95_04
mask_pos = df6_95_04_trend['import_trend'] != np.inf
mask_pos
mask_neg = df6_95_04_trend['import_trend'] != -np.inf
mask_neg
df6_95_04_trend[~mask_neg]

df6_95_04_trend.loc[~mask_pos, 'import_trend'] = df6_95_04_trend.loc[mask_pos, 'import_trend'].max()
df6_95_04_trend.loc[~mask_neg, 'import_trend'] = df6_95_04_trend.loc[mask_neg, 'import_trend'].min()

# impute import inf/-inf with max/min trend for 05_14
mask_pos = df6_05_14_trend['import_trend'] != np.inf
mask_pos
mask_neg = df6_05_14_trend['import_trend'] != -np.inf
mask_neg
df6_05_14_trend[~mask_neg]

df6_05_14_trend.loc[~mask_pos, 'import_trend'] = df6_05_14_trend.loc[mask_pos, 'import_trend'].max()
df6_05_14_trend.loc[~mask_neg, 'import_trend'] = df6_05_14_trend.loc[mask_neg, 'import_trend'].min()

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
# normalize exports on total exports of country by year
cols = ['location_id','product_id','year','export_value','export_total','export_period1','export_period2','export_trend']
train = train.copy(deep=True)
train = train[cols]
train

test = test[cols]
test = test.copy(deep=True)
test

#%%
# calculate product percent of total exports
train['export_pct'] = (train['export_value']/train['export_total'])
train.head()

test['export_pct'] = (test['export_value']/test['export_total'])
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
'''
try:
    start = time.perf_counter()
    temp = train.groupby(['location_id','year']).apply(norm_minmax, targets='export_pct').to_frame().reset_index().rename('export_pct':'export_pct_norm')
    end = time.perf_counter()
finally:
    print('run time: ', end-start)
    temp
'''

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

train_pct_norm
train_trend_std
train_trend_std.describe()
train_pct_std
test_trend_norm.describe()



#%%

# merge the pct and trend norms in
train_temp = train.join([train_pct_norm['export_pct_norm'], train_pct_std['export_pct_std'], train_trend_norm['export_trend_norm'], train_trend_std['export_trend_std']])
#train_temp = pd.merge(train, train_pct_norm['export_pct_norm'], train_pct_std['export_pct_std'], train_trend_norm['export_trend_norm'], left_index=True, right_index=True)
train = train_temp
train

test_temp = test.join([test_trend_norm['export_trend_norm'], test_trend_std['export_trend_std']])
#test_temp = pd.merge(test, test_pct_norm['export_pct_norm'], train_pct_std['export_pct_std'], test_trend_norm['export_trend_norm'], left_index=True, right_index=True)
test = test_temp
test


#df6_train['export_pct_norm'] = (df6_train['export_pct']-df6_train['export_pct'].min())/(df6_train['export_pct'].max()-df6_train['export_pct'].min())

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
for k, v in clean.items():
    try:
        if k == 'train':
            v.to_hdf('data/processed/data_clean.h5', key=k, mode='w')
        else:
            v.to_hdf('data/processed/data_clean.h5', key=k)
    except NotImplementedError:
        if k == 'train':
            v.to_hdf('data/processed/data_clean.h5', key=k, mode='w', format='t')
        else:
            v.to_hdf('data/processed/data_clean.h5', key=k, format='t')

data_clean = pd.HDFStore('data/processed/data_clean.h5', mode='r')
data_clean.keys()

#pd.read_hdf('data.h5')
train = data_clean.get('/train')
test = data_clean.get('/test')

train.head()
test.head()

data_clean.close()

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

data_prep = pd.HDFStore('data/preprocessed/data_prep.h5', mode='r')
data_prep.keys()

data_prep.close()


'''
train.to_hdf('data_clean.h5', key='train', mode='w')
test.to_hdf('data_clean.h5', key='test')
df_country.to_hdf('data_clean.h5', key='df_country', format='t')
df4_lookback.to_hdf('data_clean.h5', key='df4_lookback', format='t')
df_country_lookback.to_hdf('data_clean.h5', key='df_country_lookback', format='t')
df6_classes.to_hdf('data_clean.h5', key='df6_classes')
df_locations.to_hdf('data_clean.h5', key='df_locations')
'''

#%%

# TESTING 123

# FAIL-SAFE v0
train[train['year'] == 1995]
test[test['year'] == 2005]

#%%
# visualize the data

train.describe()


train['export_value'].hist(bins=10)
mask = train['export_pct'] == 0
train[~mask]['export_pct'].hist(bins=10)


'''
#rng = np.random.RandomState(10)  # deterministic random data
rng = train[~mask]['export_value']
#a = np.hstack((rng.normal(size=1000),
#                rng.normal(loc=5, scale=2, size=1000)))
plt.hist(rng, bins='auto')  # arguments are passed to np.histogram
plt.title("Histogram with 'auto' bins")
plt.show()
'''

#%%
# Define model

# df6, for 2007
n_countries = len(train['location_id'].unique())
n_countries

# for df6, for 2007
n_products = len(train['product_id'].unique()) * len(train['year'].unique())
n_products

n_latent_factors = 5

#%%
"""## Create Dot Product Model - Simple Shallow Learning"""

# Creating product embedding path
product_input = Input(shape=[1], name='Product_Input')
product_embedding = Embedding(n_products+1, n_latent_factors, name='Product_Embedding')(product_input)
product_vec = Flatten(name='Flatten-Products')(product_embedding)
print(product_input, product_embedding, product_vec)

#%%
# Creating country embedding path
country_input = Input(shape=[1], name='Country-Input')
country_embedding = Embedding(n_countries+1, n_latent_factors, name='Country_Embedding')(country_input)
country_vec = Flatten(name='Flatten-Countries')(country_embedding)
print(country_input, country_embedding, country_vec)

#%%
# Performing dot product and creating model; can change decay to 1e-6
prod = Dot(name='Dot_Product', axes=1)([product_vec, country_vec])
model_dot = Model([country_input, product_input], prod)
adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0, amsgrad=False, clipnorm=1)
model_dot.compile(optimizer=adam, loss='mean_squared_error', metrics=['mean_squared_error','cosine_proximity'])
model_dot.summary()

#%%
# Run the dot product model
if os.path.exists('models/regression_model_dot.h5'):
  model_dot = load_model('models/regression_model_dot.h5')
else:
  history_dot = model_dot.fit([train.location_id, train.product_id], train.export_pct_norm, batch_size=128, epochs=5, verbose=1)
  model_dot.save('models/regression_model_dot')
  plt.plot(history_dot.history['loss'])
  plt.xlabel('Epochs')
  plt.ylabel('Training Error')

# regularization loss - l2 - worth forcing the weights to stay small; some weight is going

#%%
"""https://stackoverflow.com/questions/37232782/nan-loss-when-training-regression-network

Loss probably not working because of exploding gradient problem?
"""

#%%
# Evaluate model_dot on 2008
model_dot.evaluate([test.location_id, test.product_id], test.export_pct_norm)

#%%
# Make predictions using model_dot - probably need to un-normalize from minmax
predictions_dot = model_dot.predict([test.location_id.head(10), test.product_id.head(10)])

# Denormalize - different denormalization needed depending on target used
# for i in range(0,10):
#     predictions_dot[i] = predictions_dot[i]*(test['export_value'].max()-test['export_value'].min())+test['export_value'].min()

predictions_dot

#%%
# Compare predictions with actual
[print(predictions_dot[i], test.export_pct_norm.iloc[i]) for i in range(0,10)]

'''
# Evaluate model_dot on 2017
model_dot.evaluate([df6_2017_norm.location_id, df6_2017_norm.product_id], df6_2017_norm.export_value)
'''

'''
# Make predictions using model_dot - probably need to un-normalize from minmax
predictions_dot = model_dot.predict([df6_2017_norm.location_id.head(10), df6_2017_norm.product_id.head(10)])

# Denormalize
for i in range(0,10):
    predictions_dot[i] = predictions_dot[i]*(df6_2017['export_value'].max()-df6_2017['export_value'].min())+df6_2017['export_value'].min()

predictions_dot

# Compare predictions with actual
[print(predictions_dot[i], df6_2017.export_value.iloc[i]) for i in range(0,10)]
'''

#%%
"""## Creating Neural Network"""

# Creating product embedding path
product_input = Input(shape=[1], name='Product-Input')
product_embedding = Embedding(n_products+1, n_latent_factors, name='Product-Embedding')(product_input)
product_vec = Flatten(name='Flatten-Products')(product_embedding)

#%%
# Creating country embedding path
country_input = Input(shape=[1], name='Country-Input')
country_embedding = Embedding(n_countries+1, n_latent_factors, name='Country-Embedding')(country_input)
country_vec = Flatten(name='Flatten-Countries')(country_embedding)

#%%
'''
# Compile model
# can add regularization or dropout? kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01) - leads to excessively slow learning/very high loss

# too many 0's with relu activation - try tanh or LeakyReLU(0.3); softmax for probability

# Concatenate features
conc = Concatenate()([product_vec, country_vec])

# Add fully-connected layers
fc1 = Dense(128)(conc)
fc2 = advanced_activations.LeakyReLU(alpha=0.3)(fc1)
fc3 = Dense(32)(fc2)
fc4 = advanced_activations.LeakyReLU(alpha=0.3)(fc3)
out = Dense(1)(fc4)

# Create model and compile it
model_nn = Model([country_input, product_input], out)
model_nn.compile('adam', 'mean_squared_error', metrics=['cosine_proximity'])
'''

#%%
# Compile model
# can add regularization or dropout? kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01) - leads to excessively slow learning/very high loss

# too many 0's with relu activation - try tanh or LeakyReLU(0.3); softmax for probability

# Concatenate features
conc = Concatenate()([product_vec, country_vec])

# Add fully-connected layers
fc1 = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01))(conc)
fc2 = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01))(fc1)
out = Dense(1)(fc2)

# Create model and compile it
model_nn = Model([country_input, product_input], out)
model_nn.compile('adam', 'mean_squared_error', metrics=['mean_squared_error','cosine_proximity'])
model_nn.summary()

#%%
# Run the NN model
if os.path.exists('models/regression_model_nn.h5'):
  model_nn = load_model('models/regression_model_nn.h5')
else:
  history_nn = model_nn.fit([train.location_id, train.product_id], train.export_pct_norm, epochs=5, verbose=1)
  model_nn.save('models/regression_model_nn.h5')
  plt.plot(history_nn.history['loss'])
  plt.xlabel('Number of Epochs')
  plt.ylabel('Training Error')

#%%
# Evaluate model_nn on 2008
model_nn.evaluate([test.location_id, test.product_id], test.export_pct_norm)

#%%
# Make predictions using model_nn
predictions_nn = model_nn.predict([test.location_id.head(10), test.product_id.head(10)])

# Denormalize - different denormalization needed depending on target used
# for i in range(0,10):
#     predictions_nn[i] = predictions_nn[i]*(test['export_value'].max()-test['export_value'].min())+test['export_value'].min()

predictions_nn

# Compare predictions with actual
[print(predictions_nn[i], test.export_pct_norm.iloc[i]) for i in range(0,10)]

'''
# Evaluate model_nn on 2017
model_nn.evaluate([df6_2017_norm.location_id, df6_2017_norm.product_id], df6_2017_norm.export_value)
'''

'''
# Make predictions using model_nn
predictions_nn = model_nn.predict([df6_2017_norm.location_id.head(10), df6_2017_norm.product_id.head(10)])

# Denormalize
for i in range(0,10):
    predictions_nn[i] = predictions_nn[i]*(df6_2017['export_value'].max()-df6_2017['export_value'].min())+df6_2017['export_value'].min()

predictions_nn

# Compare predictions with actual
[print(predictions_nn[i], df6_2017.export_value.iloc[i]) for i in range(0,10)]
'''

"""#Visualizing Embeddings
Embeddings are weights that are learned to represent some specific variable like products and countries in our case and, therefore, we can not only use them to get good results on our problem but also extract insight about our data.
"""

#%%
# Extract embeddings
product_em = model_nn.get_layer('Product-Embedding')
product_em_weights = product_em.get_weights()[0]

product_em_weights[:5]

#%%
pca = PCA(n_components=2)
pca_result = pca.fit_transform(product_em_weights)
sns.scatterplot(x=pca_result[:,0], y=pca_result[:,1])

#%%
product_em_weights = product_em_weights / np.linalg.norm(product_em_weights, axis=1).reshape((-1,1))
product_em_weights[0][:10]
np.sum(np.square(product_em_weights[0]))

#%%
pca = PCA(n_components=2)
pca_result = pca.fit_transform(product_em_weights)
sns.scatterplot(x=pca_result[:,0], y=pca_result[:,1])

#%%
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tnse_results = tsne.fit_transform(product_em_weights)

#%%
sns.scatterplot(x=tnse_results[:,0], y=tnse_results[:,1])

"""#Making Recommendations"""
#%%
len(train.product_id)

# Creating dataset for making recommendations for the first country
product_data = np.array(list(set(train.product_id)))
product_data

#%%
len(product_data)

train.loc[train['product_id']==8192].head()

country = np.array([1 for i in range(len(product_data))])
country[:5]

#%%
# show normalized prediction values
predictions = model_nn.predict([country, product_data])
predictions

#%%
predictions = np.array([a[0] for a in predictions])
predictions

# denormalize prediction values - different needed
# for i in range(len(predictions)):
#     predictions[i] = predictions[i]*(df6_2007['export_value'].max()-df6_2007['export_value'].min())+df6_2007['export_value'].min()

predictions

#%%
len(predictions)
predictions.min()

# show recommended products (i.e. top export values)
recommended_product_ids = (-predictions).argsort()[:5]
recommended_product_ids

#%%
# print predicted export_value - normalized
predictions[recommended_product_ids]

#%%
# show predicted product details - first all products
df6_classes = df6_classes.reset_index()
df6_classes.head()

#%%
df6_classes[df6_classes['index'].isin(recommended_product_ids)]

#%%
# KEY
df6_classes.iloc[recommended_product_ids]

#%%
product_data = np.array(list(train.product_id))
product_data

product_data[0:20] #10 years = 1995-2004

product_data

#%%
