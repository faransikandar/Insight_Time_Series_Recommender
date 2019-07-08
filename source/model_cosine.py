#%%
# inspired from https://www.analyticsvidhya.com/blog/2018/06/comprehensive-guide-recommendation-engine-python/

#%%
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
from sklearn.metrics.pairwise import pairwise_distances

from source.data_load import *

#%%
# Load the data
dict_data = data_load()
dict_data

data_clean = dict_data['data_clean']
data_clean.keys()
test = data_clean.get('/test')
train = data_clean.get('train')

#%%
# Define model inputs
n_countries = len(train['location_id'].unique())
n_products = train['product_id'].max()
n_years = train['year'].max()
n_years = 2017
n_trends = len(train['export_trend_norm'].unique()) #int(round(train['export_trend_std'].max()))
#n_trends = 10
n_latent_factors = 10

print(n_countries)
print(n_products)
print(n_years)
print(n_trends)

#%%
# If want to simplify to one year analysis

train_1995 = train[train['year'] == 1995].reset_index()
test_2005 = test[test['year'] == 2005].reset_index()

train_1995 = train_1995.drop(columns=['index'],axis=1)
test_2005 = test_2005.drop(columns=['index'],axis=1)

# Simplify to non-zero inputs
mask = train['export_value'] <= 0
mask_1995 = train_1995['export_value'] <= 0

# Sparse dfs
train_sparse = train[~mask]
train_1995_sparse = train_1995[~mask]
train_sparse.head()

#%%
# Small dataset
train_small = train_1995[:10]
train_small

#%%
# Redefine for cosine similarity model
n_countries = len(train_1995['location_id'].unique())
n_products = len(train_1995['product_id'].unique())

print(n_countries)
print(n_products)

#%%
data_matrix = np.zeros((n_countries, n_products))
for i in train_small.itertuples():
    print(i[1]-1, i[2], i[3])

#%%
