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
from keras.callbacks import ModelCheckpoint, History, ReduceLROnPlateau
from keras.constraints import maxnorm
from keras.layers import advanced_activations, Concatenate, Dense, Dot, Dropout, Embedding, Flatten, Input, LSTM, Reshape
from keras.models import load_model, Model, Sequential
from keras.utils import np_utils

from sklearn.model_selection import train_test_split

from source.data_load import *

#%%
def define_model(dict_data_key_clean, model_name, history_name, model_load=True, gdrive=False):

    # Load the data
    print('Defining the data...')

    dict_data = data_load()
    dict_data

    data_clean = dict_data[dict_data_key_clean] # make this the target in the def
    data_clean.keys()
    test = data_clean.get('/test')
    train = data_clean.get('/train')

    # Define model inputs
    n_countries = len(train['location_id'].unique())
    n_products = train['product_id'].max()
    n_years = train['year'].max()
    n_years = 2017
    n_trends = len(train['export_trend_norm'].unique()) #int(round(train['export_trend_std'].max()))
    n_latent_factors = 10

    print(n_countries)
    print(n_products)
    print(n_years)
    print(n_trends)

    # Create embeddings
    print('Creating embeddings...')

    # Creating product embedding path
    product_input = Input(shape=[1], name='Product-Input')
    product_embedding = Embedding(n_products+1, n_latent_factors, name='Product-Embedding')(product_input)
    product_vec = Flatten(name='Flatten-Products')(product_embedding)

    # Creating country embedding path
    country_input = Input(shape=[1], name='Country-Input')
    country_embedding = Embedding(n_countries+1, n_latent_factors, name='Country-Embedding')(country_input)
    country_vec = Flatten(name='Flatten-Countries')(country_embedding)

    # Creating year embedding path
    year_input = Input(shape=[1], name='Year-Input')
    year_embedding = Embedding(n_years+1, n_latent_factors, name='Year-Embedding')(year_input)
    year_vec = Flatten(name='Flatten-Years')(year_embedding)

    # Creating trend embedding path
    trend_input = Input(shape=[1], name='Trend-Input')
    trend_embedding = Embedding(n_trends+1, n_latent_factors, name='Trend-Embedding')(trend_input)
    trend_vec = Flatten(name='Flatten-Trends')(trend_embedding)

    # Define alternative loss metrics

    # From: https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html
    # Typically used for regression. It’s less sensitive to outliers than the MSE as it treats error as square only inside an interval.

    def Huber(yHat, y, delta=1.):
        return np.where(np.abs(y-yHat) < delta,.5*(y-yHat)**2 , delta*(np.abs(y-yHat)-0.5*delta))

    # Full neural network (NN) model - including multiple layers, embeddings, and engineered features
    # Compile NN model

    # Inspired from: http://digital-thinking.de/deep-learning-combining-numerical-and-text-features-in-deep-neural-networks/

    print('Definining inputs, layers, and hyperparameters...')

    trend_rank_input = Input(shape=[1], name='Trend-Rank-Input')
    trend_class_input = Input(shape=[1], name='Trend-Class-Input')

    # Concatenate categorical features
    dot_cat = Dot(name='Dot_Product', axes=1)([country_vec, product_vec])
    conc_cat = Concatenate(name='Concatentate-Country-Product')([country_vec, product_vec])

    # Add fully-connected layers
    fc1 = Dropout(0.3, name='Dropout-1')(conc_cat)
    fc2 = Dense(128, activation='relu', kernel_initializer='glorot_uniform', kernel_constraint=maxnorm(3), name='Dense-1')(fc1)
    fc3 = Concatenate(name='Concatenate-FC-Class-Rank')([fc2, trend_class_input, trend_rank_input])
    fc4 = Dropout(0.3, name='Dropout-2')(fc3)
    fc5 = Dense(32, activation='relu', kernel_initializer='glorot_uniform', kernel_constraint=maxnorm(3), name='Dense-2')(fc4)
    out = Dense(1, name='Output')(fc5)

    os.getcwd()

    if model_load == False:
    # Set up path for saving model and history
        if gdrive == True:
            filepath_model = '/content/gdrive/My Drive/Colab Notebooks/Insight_Net_Align/models/' + model_name + '-{epoch:02d}-{loss:.4f}.hdf5'
            filepath_history = '/content/gdrive/My Drive/Colab Notebooks/Insight_Net_Align/models/' + history_name
        else:
            directory = os.path.dirname(os.path.abspath(model_name))
            filepath_model = os.path.join(directory,'models/',model_name + '-{epoch:02d}-{loss:.4f}.hdf5')
            filepath_history = os.path.join(directory,'models/',history_name)

        # Define the model callbacks
        checkpoint = ModelCheckpoint(filepath_model, monitor='loss', verbose=1, save_best_only=True, mode='min')
        history = History()
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
        callbacks_list = [checkpoint, history, reduce_lr]

    elif model_load == True:
        print('Loading the model...')

        if gdrive == True:
            filepath_model = '/content/gdrive/My Drive/Colab Notebooks/Insight_Net_Align/models/' + model_name
            filepath_history = '/content/gdrive/My Drive/Colab Notebooks/Insight_Net_Align/models/' + history_name
        else:
            directory = os.path.dirname(os.path.abspath(model_name))
            filepath_model = os.path.join(directory,'models/',model_name)
            filepath_history = os.path.join(directory,'models/',history_name)

        model_nn = load_model(filepath_model)
        history_nn = pickle.load(open(filepath_history, 'rb'))
        history_nn

    # Create model and compile it
    model_nn = Model([country_input, product_input, trend_class_input, trend_rank_input], out)

    sgd = optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
    nadam = optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=2e-4, amsgrad=False, clipnorm=1) # clipvalue = 0.5 or 0.25 for exploding gradients

    # Compile the model
    print('Compiling the model...')

    model_nn.compile(optimizer=adam, loss='mean_squared_error', metrics=['logcosh', 'mean_absolute_error', 'mean_squared_error','cosine_proximity'])
    print(model_nn.summary())

    # Train a model
    if model_load == False:
        print('Training the model...')

        history_nn = model_nn.fit( [train.location_id, train.product_id, train.export_trend_class, train.export_trend_pct_rank], train.export_val_std_all,
                        batch_size=32, epochs=10, verbose=1, callbacks = callbacks_list )

        with open(filepath_history, 'wb') as file_pi:
            pickle.dump(history_nn, file_pi)

    # Plot the history
    print('Plotting training history...')

    plt.plot(history_nn.history['loss'])
    plt.xlabel('Number of Epochs')
    plt.ylabel('Training Error')

    # Evaluate model_nn
    print('Evaluating the model...')
    print(model_nn.metrics_names)

    model_eval = model_nn.evaluate([test.location_id, test.product_id, test.export_trend_class, test.export_trend_pct_rank], test.export_val_std_all)
    print(model_eval)

def main():
    dict_data_key_clean = 'data_example_clean'
    model_name = 'model_nn_full_example-10-0.2273.hdf5'
    history_name = 'history_nn_full_example'
    define_model(dict_data_key_clean, model_name, history_name, model_load=True)

if __name__ == "__main__":
    main()
