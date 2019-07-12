
import os
import sys
import time
import warnings
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

from collections import Counter, defaultdict, OrderedDict
from itertools import chain
from keras.callbacks import ModelCheckpoint
from keras.models import load_model, Model, Sequential
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, mean_absolute_error

from source.data_load import *


def make_predictions(dict_data_key_clean, dict_data_key_raw, model_name, history_name, predictions_key, country_choice):

    # Load the data
    dict_data = data_load()
    dict_data

    data_clean = dict_data[dict_data_key_clean]
    data_clean.keys()
    test = data_clean.get('/test')
    train = data_clean.get('/train')

    test.head()

    # Define df_locations df
    data_example = dict_data[dict_data_key_raw]
    data_example.keys()

    df_locations = data_example.get('/df_locations')
    df_locations
    df_classes = data_example.get('/df_classes')
    df_classes



    # Create baseline df for comparison of pct change
    baseline = pd.DataFrame()

    # Add in values from test and train to calculate change from zero, once include predictions - if not merging, order matters for first col bc that will define index

    baseline = pd.merge(train['export_period_total'], test['export_period_total'], how='right', left_index=True, right_index=True)
    baseline = baseline.rename(index=str, columns={'export_period_total_x':'export_period_train', 'export_period_total_y':'export_period_test'})
    baseline

    baseline.describe()


    # Calculate pct change
    baseline['export_pct_change'] = (baseline['export_period_test']-baseline['export_period_train'])/baseline['export_period_train']
    baseline


    # Clean the NaNs in export_pct_change
    mask_nan = baseline['export_pct_change'].isnull() # returns values which ARE NaN
    baseline[mask_nan]

    baseline['export_pct_change'] = np.where(mask_nan, baseline['export_period_test'], baseline['export_pct_change'])
    baseline[mask_nan]
    # baseline.loc[baseline['export_pct_change'].isnull(), baseline['export_pct_change']] = baseline['export_period_test'] # same as

    # Clean the NaNs in export_period in export_period_train
    mask_nan = baseline['export_period_train'].isnull()
    baseline[mask_nan]

    baseline['export_period_train'] = np.where(mask_nan, 0, baseline['export_period_train'])
    baseline[mask_nan]

    # Check for nulls
    baseline.isnull().values.any()


    # Clean the inf/-inf values
    mask_pos = baseline['export_pct_change'] != np.inf # returns values which are NOT inf
    mask_neg = baseline['export_pct_change'] != -np.inf # returns values which are NOT -inf

    baseline[~mask_pos]
    baseline[~mask_neg]

    baseline['export_pct_change'] = np.where(~mask_pos, baseline['export_period_test'], baseline['export_pct_change'])
    baseline[~mask_pos]

    baseline['export_pct_change'] = np.where(~mask_neg, -baseline['export_period_test'], baseline['export_pct_change'])
    baseline[~mask_neg]

    baseline


    # Create metric to measure NEW goods - can also do -1 (decrease), 0 (no change), 1 (increase), 2 (new good)
    mask_new = (baseline['export_period_test'] > 0) & (baseline['export_period_train'] == 0)
    baseline[mask_new]

    baseline['export_pct_change_rank'] = baseline['export_pct_change'].rank(method='min', ascending=False)

    mask_decrease = (baseline['export_pct_change'] < 0)
    mask_stagnant = (baseline['export_pct_change'] == 0)
    mask_increase = (baseline['export_pct_change'] > 0) & (baseline['export_period_train'] != 0)

    baseline['export_class'] = np.select([mask_new, mask_decrease, mask_stagnant, mask_increase], [2, -1, 0, 1], default=np.nan)

    # baseline['export_class'] = np.where(mask_new, 2, np.nan) # same as above
    # baseline['export_class'] = np.where(mask_decrease, -1, baseline['export_class'])
    # baseline['export_class'] = np.where(mask_stagnant, 0, baseline['export_class'])
    # baseline['export_class'] = np.where(mask_increase, 1, baseline['export_class'])

    baseline


    # DATA LEAKAGE RISK IF TEST ON TEST TREND VALUES - need to create train.export_trend_pct_rank standin for test
    # (basically like projecting time trends forward, but it's nice, because you can toy with these and generate them however you like to run scenario analysis)

    # Create train placeholder set for test export_trend_pct_rank
    train_forecast = pd.concat([train['export_trend_pct_rank'], test['export_trend_pct_rank']], axis=1)

    # Impute missing values with mean
    train_forecast = train_forecast.fillna(train['export_trend_pct_rank'].mean())

    # Just return the train column (1st)
    train_forecast.columns[1]
    train_forecast = train_forecast.iloc[:,0:1]
    train_forecast

    # Create train placeholder for export_trend_class
    train_forecast = pd.merge(train_forecast, train['export_trend_class'], how='left', left_index=True, right_index=True)
    train_forecast['export_trend_class'] = train_forecast['export_trend_class'].fillna(0)
    train_forecast


    # Define the model filepath
    os.getcwd()

    directory = os.path.dirname(os.path.abspath('model_nn_full2-10-0.7100.hdf5'))
    directory
    model_filename = os.path.join(directory,'models/model_nn_full2-10-0.7100.hdf5')
    model_filename
    history_filename = os.path.join(directory,'models/history_nn_full2')
    history_filename


    # Load the model
    model_nn_full = load_model(model_filename)
    model_nn_full.summary()

    # Load the history
    history_nn_full = pickle.load(open(history_filename, 'rb'))
    history_nn_full


    # Plot the history of the model
    plt.plot(history_nn_full['loss'])
    plt.xlabel('Number of Epochs')
    plt.ylabel('Training Error')


    # Evaluate model - can set = and add callbacks in order to get history?
    print(model_nn_full.metrics_names)
    model_nn_full.evaluate([test.location_id, test.product_id, train_forecast.export_trend_class, train_forecast.export_trend_pct_rank], test.export_val_std_all) # takes 2.5 min?? on CPU test loss of 0.75 GOOD!



    # Show standardized prediction values - all countries, products, years
    predictions_raw = model_nn_full.predict([test.location_id, test.product_id, train_forecast.export_trend_class, train_forecast.export_trend_pct_rank]) #year, trend for full model
    predictions_raw


    # Hard code std and mean values for loop
    test_export_value_std = test['export_value'].std()
    test_export_value_mean = test['export_value'].mean()

    # De-standardize predictions
    predictions = []
    for i in range(len(predictions_raw)):
        predictions.append((predictions_raw[i] * test_export_value_std) + test_export_value_mean)

    predictions[:20]
    len(predictions)
    type(predictions)
    # currently a list of arrays
    predictions[0:5]


    # concatenate into a single array
    predictions = np.concatenate(predictions, axis=0)

    # Merge predictions into baseline
    baseline['predictions'] = predictions
    baseline


    # Calculate pct change
    baseline['pred_pct_change'] = (baseline['predictions'] - baseline['export_period_train']) / baseline['export_period_train']
    baseline


    # Clean the NaNs in export_pct_change
    mask_nan = baseline['pred_pct_change'].isnull() # returns values which ARE NaN
    baseline[mask_nan]

    baseline['pred_pct_change'] = np.where(mask_nan, baseline['predictions'], baseline['pred_pct_change'])
    baseline[mask_nan]
    # baseline.loc[baseline['export_pct_change'].isnull(), baseline['export_pct_change']] = baseline['export_period_test'] # same as

    # Check for nulls
    baseline.isnull().values.any()


    # Clean the inf/-inf values
    mask_pos = baseline['pred_pct_change'] != np.inf # returns values which are NOT inf
    mask_neg = baseline['pred_pct_change'] != -np.inf # returns values which are NOT -inf

    baseline[~mask_pos]
    baseline[~mask_neg]

    baseline['pred_pct_change'] = np.where(~mask_pos, baseline['predictions'], baseline['pred_pct_change'])
    baseline[~mask_pos]

    baseline['pred_pct_change'] = np.where(~mask_neg, baseline['predictions'], baseline['pred_pct_change'])
    baseline[~mask_neg]


    # Merge train and test countries, products, years
    cols = ['location_id', 'product_id', 'year']
    train_ref = train[cols]
    test_ref = test[cols]

    train_ref.shape
    test_ref.shape

    df_reference = pd.merge(train_ref, test_ref, how = 'right', on = ['location_id', 'product_id', 'year'])
    df_reference

    #Merge df_classes into reference table
    df_reference = pd.merge(df_reference, df_classes[['index', 'name']], how = 'left', left_on = 'product_id', right_on = 'index')
    df_reference = df_reference.drop(columns='index')
    df_reference

    # ensure that the index dtypes are the same
    baseline.index = baseline.index.astype('int64')
    df_reference.index = df_reference.index.astype('int64')


    # Merge reference and baseline dfs
    df_predictions = pd.concat([df_reference, baseline], axis = 1)
    df_predictions.shape
    df_predictions
    df_predictions.describe()


    # Which country to make prediction for
    df_locations[df_locations['name_en'] == country_choice]

    location = int(df_locations[df_locations['name_en'] == country_choice]['index'])
    location

    # Return recommendations for a given country
    df_recs = df_predictions.loc[(df_predictions['location_id'] == location) & (df_predictions['year'] == 2014)]

    # Return the nlargest pct change
    top_recs = df_recs.nlargest(50, columns = 'pred_pct_change')
    top_recs


    # Calculate rank for actual export change just for this country
    top_recs['export_pct_change_rank_loc'] = top_recs['export_pct_change_rank'].rank(method='min', ascending=False)


    # Return only select columns
    top_recs_display = top_recs[['product_id','name','export_period_test','export_pct_change','export_pct_change_rank_loc','predictions','pred_pct_change']]
    print( 'These are your top recommendations for country', country_choice, 'for the period 2005-2014!!!',
            '(Products appear in descending order, with highest predicted percent growth at top of list.)' )
    print("\n#####################################\n")
    print(top_recs_display)

    # top_recs.drop(['year','location_id','export_period_train','export_class'], axis=1)


    # Redefine problem to only include change from 0

    # Check what these products are
    df_predictions[df_predictions['export_period_train'] == 0].shape

    # Redefine the recommendation df
    df_recs_new = ( df_predictions.loc[(df_predictions['location_id'] == location) & (df_predictions['year'] == 2014) &
                    (df_predictions['export_period_train'] == 0)] )

    df_recs_new
    # Make recs for new products
    df_recs_new.nlargest(20, columns = 'pred_pct_change', keep = 'first')


    # Validation metrics

    # Calculate cosine_similarity

    print(cosine_similarity([df_predictions['export_period_test']], [df_predictions['predictions']]))
    print(cosine_similarity([df_predictions['export_pct_change']], [df_predictions['pred_pct_change']]))

    print(mean_absolute_error([df_predictions['export_pct_change']], [df_predictions['pred_pct_change']]))
    print(mean_squared_error([df_predictions['export_pct_change']], [df_predictions['pred_pct_change']]))


    # Next step is to rank the pct_change actual vs. predict and do MSE validation
    df_predictions['export_pct_change_rank'] = df_predictions['export_pct_change'].rank(ascending=False, method='min')
    df_predictions.sort_values(by=['export_pct_change_rank'])

    # REALLY useful - shows the top predicted export pct change areas - travel/tourism, computer inputs, etc. dominate
    df_predictions['pred_pct_change_rank'] = df_predictions['pred_pct_change'].rank(ascending=False, method='min')
    df_predictions.sort_values(by=['pred_pct_change_rank'])



    # Calculate cosine similarity and MSE between export_pct and pred_pct ranks

    # See if there's much difference between train and test export values
    print(cosine_similarity([df_predictions['export_period_train']], [df_predictions['export_period_test']]))

    # Cosine for predicted pct change
    print(cosine_similarity([df_predictions['export_pct_change_rank']], [df_predictions['pred_pct_change_rank']]))

    # This is the one that really matters - otherwise cosine similarity is more susceptible to minor perturbations - this best takes into account order
    df_predictions['export_period_train_rank'] = df_predictions['export_period_train'].rank(ascending=False, method='min')
    df_predictions['export_period_test_rank'] = df_predictions['export_period_test'].rank(ascending=False, method='min')
    print(cosine_similarity([df_predictions['export_period_train_rank']], [df_predictions['export_period_test_rank']]))

    # MSE's - hard to interpret because so large
    print(mean_absolute_error([df_predictions['export_pct_change_rank']], [df_predictions['pred_pct_change_rank']]))
    print(mean_squared_error([df_predictions['export_pct_change_rank']], [df_predictions['pred_pct_change_rank']]))

    # Save df to hdf5
    df_predictions.to_hdf('data/processed/data_predictions.h5', key=predictions_key)

    return top_recs_display

def main():

    dict_data_key_clean = 'data_example_clean'
    dict_data_key_raw = 'data_example'
    model_name = 'model_nn_full_example-10-0.2273.hdf5'
    history_name = 'history_nn_full_example'
    predictions_key = 'df_predictions_nn_full_example'

    # Pick from a list of countries?
    print('Pick from this list of locations!')
    df_locations = data_load()['data_example'].get('/df_locations')
    print(df_locations['name_en'])
    print('#########################################')
    country_choice = input('What country do you want to predict exports for? ')
    make_predictions(dict_data_key_clean, dict_data_key_raw, model_name, history_name, predictions_key, country_choice)

if __name__ == "__main__":
    main()
