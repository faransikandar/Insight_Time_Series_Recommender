#%%
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from keras import optimizers, regularizers
from keras.callbacks import ModelCheckpoint, History, ReduceLROnPlateau
from keras.constraints import maxnorm
from keras.layers import advanced_activations, Concatenate, Dense, Dot, Dropout, Embedding, Flatten, Input, LSTM, Reshape
from keras.models import load_model, Model, Sequential
from keras.utils import np_utils
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, mean_absolute_error

# acrobatics for matplotlib on Mac OSX
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

from source.data_load import *

#%%
def make_predictions( dict_data_key_raw, dict_data_key_clean, model_name, full_features, history_name,
    predictions_key, n_predictions, country_choice ):
    '''
    Inputs:
    dict_data_key_raw: 'data_full' or 'data_example' - include raw data files for 6, digit, 2 digit, or sample trade data, product classes, locations
    dict_data_key_clean: 'data_full_clean' or 'data_example_clean' - include train and test (calculated based on what is called in 'raw' above)
    model_name: name of model file, including file extension
    full_features: boolean, True if all time features used; False if only country + product inputs used
    history_name: name of history file
    predictions_key: name of key which you want to save predictions to within data_predictions.h5 file
    n_predictions: number of top recommendations to display
    country_choice: the country to make recommendations for, for the 2005-2014 period (based on 1995-2004 training) (fed as user input in main)
    ________________________
    Outputs:
    top_recs_display: display of top n predictions, ranked in descending order based on highest predicted growth rate (pandas dataframe)
    ________________________
    '''
    # Load the data
    dict_data = data_load()
    dict_data

    data_clean = dict_data[dict_data_key_clean]
    data_clean.keys()
    test = data_clean.get('/test')
    train = data_clean.get('/train')

    test.head()

    # Define df_locations df
    data_raw = dict_data[dict_data_key_raw]
    data_raw.keys()

    if dict_data_key_raw == 'data_full':
        df_locations = data_raw.get('/classifications/location')
        df_locations
        df_classes = data_raw.get('/classifications/hs_product')
        df_classes = df_classes[df_classes['level']=='6digit']

    elif dict_data_key_raw == 'data_sample' or dict_data_key_raw == 'data_2digit':
        df_locations = data_raw.get('/df_locations')
        df_locations
        df_classes = data_raw.get('/df_classes')
        df_classes

    print("##################################################")
    print('Calculating baselines for comparison...')
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

    directory = os.path.dirname(os.path.abspath('abcdef'))
    directory
    model_filename = os.path.join(directory, 'models', model_name)
    model_filename
    history_filename = os.path.join(directory,'models', history_name)
    history_filename


    print("##################################################")
    print('Model Summary...')
    # Load the model
    model = load_model(model_filename)
    model.summary()

    # Load the history
    history = pickle.load(open(history_filename, 'rb'))
    history

    # Plot the history of the model - some previously run history files just dicts, so need to try two methods
    # try:
    #     plt.plot(history['loss'])
    #     plt.xlabel('Number of Epochs')
    #     plt.ylabel('Training Error')
    # except:
    #     plt.plot(history.history['loss'])
    #     plt.xlabel('Number of Epochs')
    #     plt.ylabel('Training Error')

    # Evaluate model - can set = and add callbacks in order to get history?
    # print(model.metrics_names)
    # model.evaluate([test.location_id, test.product_id, train_forecast.export_trend_class, train_forecast.export_trend_pct_rank], test.export_val_std_all) # takes 2.5 min?? on CPU test loss of 0.75 GOOD!

    print("##################################################")
    print('This may take a minute. Making predictions...')

    # Show standardized prediction values - all countries, products, years
    if full_features == True:
        predictions_raw = model.predict([test.location_id, test.product_id, train_forecast.export_trend_class, train_forecast.export_trend_pct_rank]) #year, trend for full model
        predictions_raw
    elif full_features == False:
        predictions_raw = model.predict([test.location_id, test.product_id])

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
    top_recs = df_recs.nlargest(n_predictions, columns = 'pred_pct_change')
    top_recs


    # Calculate rank for actual export change just for this country
    top_recs['export_pct_change_rank_loc'] = top_recs['export_pct_change_rank'].rank(method='min', ascending=False)

    # Return only select columns
    top_recs_display = top_recs[['product_id','name','export_period_test','export_pct_change','export_pct_change_rank_loc','predictions','pred_pct_change']]

    print("##################################################")
    print('These are your top recommendations for country/territory', country_choice, 'for the period 2005-2014!!!')
    print('Products appear in descending order, with highest predicted percent growth at top of list.)')
    print("\n#####################################\n")
    print(top_recs_display)

    # top_recs.drop(['year','location_id','export_period_train','export_class'], axis=1)

    # # Redefine problem to only include change from 0 - not all countries will have this so better to exclude
    #
    # # Check what these products are
    # df_predictions[df_predictions['export_period_train'] == 0].shape
    #
    # # Redefine the recommendation df
    # df_recs_new = ( df_predictions.loc[(df_predictions['location_id'] == location) & (df_predictions['year'] == 2014) &
    #                 (df_predictions['export_period_train'] == 0)] )
    #
    # df_recs_new
    # # Make recs for new products
    # df_recs_new.nlargest(20, columns = 'pred_pct_change', keep = 'first')

    # Validation metrics

    # Calculate cosine_similarity

    print("##################################################")
    print('Calculating validation metrics for Predicted vs. Actual...')

    print( 'Cosine Similarity for Predicted vs. Actual Export Value: ',
            cosine_similarity([df_predictions['export_period_test']], [df_predictions['predictions']]) )

    print( 'Cosine Similarity for Predicted vs. Actual Export Percent Change: ',
            cosine_similarity([df_predictions['export_pct_change']], [df_predictions['pred_pct_change']]) )

    # MSE's - hard to interpret because so large
    # print(mean_absolute_error([df_predictions['export_pct_change']], [df_predictions['pred_pct_change']]))
    # print(mean_squared_error([df_predictions['export_pct_change']], [df_predictions['pred_pct_change']]))

    # Next step is to rank the pct_change actual vs. predict and do MSE validation
    df_predictions['export_pct_change_rank'] = df_predictions['export_pct_change'].rank(ascending=False, method='min')
    df_predictions.sort_values(by=['export_pct_change_rank'])

    # REALLY useful - shows the top predicted export pct change areas - travel/tourism, computer inputs, etc. dominate
    df_predictions['pred_pct_change_rank'] = df_predictions['pred_pct_change'].rank(ascending=False, method='min')
    df_predictions.sort_values(by=['pred_pct_change_rank'])


    # Cosine for predicted pct change
    # This is the one that really matters - otherwise cosine similarity is more susceptible to minor perturbations - this best takes into account order

    print( 'Cosine Similarity for Predicted vs. Actual Export Percent Change Rank: ',
            cosine_similarity([df_predictions['export_pct_change_rank']], [df_predictions['pred_pct_change_rank']]) )
    # print(mean_absolute_error([df_predictions['export_pct_change_rank']], [df_predictions['pred_pct_change_rank']]))
    # print(mean_squared_error([df_predictions['export_pct_change_rank']], [df_predictions['pred_pct_change_rank']]))

    print("##################################################")
    print('Calculating change metrics for Train vs. Test...')

    # See if there's much difference between train and test export values
    print( 'Cosine Similarity for Train vs. Test Export Value: ',
            cosine_similarity([df_predictions['export_period_train']], [df_predictions['export_period_test']]) )

    df_predictions['export_period_train_rank'] = df_predictions['export_period_train'].rank(ascending=False, method='min')
    df_predictions['export_period_test_rank'] = df_predictions['export_period_test'].rank(ascending=False, method='min')

    print( 'Cosine Similarity for Train vs. Test Export Value Rank: ',
            cosine_similarity([df_predictions['export_period_train_rank']], [df_predictions['export_period_test_rank']]) )

    # Save df to hdf5
    df_predictions.to_hdf('data/processed/data_predictions.h5', key=predictions_key) #default mode='a'

    return top_recs_display

#%%
def main():

    dict_data_key_raw = 'data_2digit'
    dict_data_key_clean = 'data_2digit_clean'
    model_name = 'model_5L_full_2digit-10-0.3169.hdf5'
    full_features = True
    history_name = 'history_5L_full_2digit'
    predictions_key = 'df_predictions_5L_full_2digit'
    n_predictions = 50

    print("##################################################")
    # Pick from a list of countries?
    print('Pick from this list of locations!')
    df_locations = data_load()['data_sample'].get('df_locations')
    print(df_locations['name_en'])
    print("##################################################")
    country_choice = input('What country/territory do you want to recommend exports for? ')

    make_predictions( dict_data_key_raw, dict_data_key_clean, model_name, full_features, history_name,
        predictions_key, n_predictions, country_choice )

    print("##################################################")
    print('Success!')
    print("##################################################")


if __name__ == "__main__":
    main()
