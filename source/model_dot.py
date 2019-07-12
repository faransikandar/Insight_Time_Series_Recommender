#%%
# inspired from https://github.com/TannerGilbert/Tutorials/blob/master/Recommendation%20System/Recommendation%20System.ipynb

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
import tensorflow as tf
from keras import optimizers, regularizers
from keras.callbacks import ModelCheckpoint
from keras.layers import advanced_activations, Concatenate, Dense, Dot, Dropout, Embedding, Flatten, Input, LSTM, Reshape
from keras.models import load_model, Model, Sequential
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

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
# Create embeddings

# Creating product embedding path
product_input = Input(shape=[1], name='Product_Input')
product_embedding = Embedding(n_products+1, n_latent_factors, name='Product_Embedding')(product_input)
product_vec = Flatten(name='Flatten-Products')(product_embedding)
print(product_input, product_embedding, product_vec)

# Creating country embedding path
country_input = Input(shape=[1], name='Country-Input')
country_embedding = Embedding(n_countries+1, n_latent_factors, name='Country_Embedding')(country_input)
country_vec = Flatten(name='Flatten-Countries')(country_embedding)
print(country_input, country_embedding, country_vec)

#%%
# Performing dot product and creating model; can change decay to 1e-6
prod = Dot(name='Dot_Product', axes=1)([product_vec, country_vec])
model_dot = Model([country_input, product_input], prod)
filepath="/content/gdrive/My Drive/Colab Notebooks/Insight_Net_Align/models/model_dot_pct_std-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
history = History()
callbacks_list = [checkpoint, history]

adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0, amsgrad=False) #clipnorm=1 or clipvalue=0.5

model_dot.compile(optimizer=adam, loss='mean_squared_error', metrics=['logcosh','mean_absolute_error','mean_squared_error','cosine_proximity'])
model_dot.summary()

#%%
# Run the dot product model
if os.path.exists('model.h5'):
  model_dot = load_model('regression_model_dot.h5')
else:
  history_dot = model_dot.fit([train.location_id, train.product_id], train.export_val_std_all, batch_size=128, epochs=20, verbose=1, callbacks=callbacks_list)
  #model_dot.save('regression_model_dot.h5')

  with open('/content/gdrive/My Drive/Colab Notebooks/Insight_Net_Align/models/history_dot_pct_std', 'wb') as file_pi:
    pickle.dump(history_dot.history, file_pi)

  plt.plot(history_dot.history['loss'])
  plt.xlabel('Epochs')
  plt.ylabel('Training Error')

#%%
# Evaluate model_dot
model_dot.evaluate([test.location_id, test.product_id], test.export_val_std_all)

#%%
# Make select predictions using model_dot - probably need to un-normalize from minmax
predictions_dot = model_dot.predict([test.location_id.head(20), test.product_id.head(20)])

[print(predictions_dot[i], test.export_pct_std.iloc[i]) for i in range(0,20)]
