'''
Insight_Net_Align

Alternative Models

# inspired from https://github.com/TannerGilbert/Tutorials/blob/master/Recommendation%20System/Recommendation%20System.ipynb
'''

#%%
from surprise import Reader, Dataset
from surprise import SVD, evaluate, accuracy, GridSearch
from surprise.model_selection import train_test_split
import os
import pandas as pd
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
# LOAD THE DATA
from source.data_load import *

dict_data = data_load()
dict_data

hdf = dict_data['hdf']
data_clean = dict_data['data_clean']

#%%
# df6_classes = data_clean.get('/df6_classes')
# df_locations = data_clean.get('/df_locations')
test = data_clean.get('/test')
train = data_clean.get('/train')

train.head()

#%%
# Surprise libraries
from surprise import KNNBasic, SVD, GridSearch, Reader, Dataset, Trainset, evaluate, accuracy
from surprise.model_selection import train_test_split
from surprise.dump import dump, load

#%%
# Prepare data for surprise

# A reader is still needed but only the rating_scale param is required - have to predefine scale, so best to use norm data
reader = Reader(rating_scale=(0.,1.))

# Create Dataset object for train
train_data = Dataset.load_from_df(train[['location_id','product_id','export_val_norm_all']], reader)
train_data

# Split data into 5 folds - only necessary if doing CV or build on full trainset
# train_data.split(n_folds=5)
train_set = train_data.build_full_trainset()
train_set

#%%
# call knn and train it
knn_algo = KNNBasic()
knn_algo.fit(train_set)

#%%
# save knn_algo
dump('models/surprise_knn.h5')

#%%
# evaluate (if doing CV - train_set must have folds
# knn_eval = evaluate(knn_algo, train_set, measures=['MAE','RMSE'])

#%%
# build test data
test_data = Dataset.load_from_df(test[['location_id','product_id','export_val_norm_all']], reader)
test_data
train_set, test_set = train_test_split(test_data, test_size=1.)

#%%
# make predictions on test set
predictions = knn_algo.test(test_set)
predictions = pd.DataFrame(test_set)
predictions

#%%
# save predictions
predictions.to_hdf('data/processed/data_predictions', key='surprise_knn')

#%%
# display predictions
predictions.drop("details", inplace=True, axis=1)
predictions.columns = ['location_id', 'product_id', 'export_val_norm_all', 'knn_predictions']
predictions.head()


# GridSearch CV
# Takes a lot of memory to optimize parameters for SVD

# param_grid = {'lr_all': [0.002, 0.005], 'reg_all': [0.4,0.6]}
# grid_search = GridSearch(SVD, param_grid, measures=['MAE','RMSE'])
# grid_search.evaluate(data)
#
# # Surprise requires train, test split
# trainset = data.build_full_trainset() #train_test_split(data, test_size=0)

# Train the model using SVD
# algo = SVD()
# algo.fit(trainset)
