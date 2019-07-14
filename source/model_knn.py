'''
Insight_Net_Align

Alternative Models

# inspired from https://github.com/TannerGilbert/Tutorials/blob/master/Recommendation%20System/Recommendation%20System.ipynb
'''

#%%
import os
import numpy as np
import pandas as pd
from surprise import Reader, Dataset
from surprise import KNNBasic, SVD, GridSearch, Reader, Dataset, Trainset, evaluate, accuracy
from surprise.model_selection import train_test_split
from surprise.dump import dump, load
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, mean_absolute_error

#%%
# LOAD THE DATA
from source.data_load import *

dict_data = data_load()
dict_data

hdf = dict_data['data_full']
data_clean = dict_data['data_full_clean']

#%%
# df6_classes = data_clean.get('/df6_classes')
# df_locations = data_clean.get('/df_locations')
test = data_clean.get('/test')
train = data_clean.get('/train')

train.head()

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
dump('models/model_surprise_knn.h5')

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

predictions.describe()

#%%
# display predictions
#predictions.drop("details", inplace=True, axis=1)
predictions.columns = ['location_id', 'product_id', 'knn_predictions']
predictions.head()

#%%
# save predictions
predictions.to_hdf('data/processed/data_predictions.h5', key='df_surprise_knn')

data_predictions = pd.HDFStore('data/processed/data_predictions.h5', mode = 'a')

data_predictions.keys()

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

#%%
#%%
# Create baseline df for comparison of pct change
baseline = pd.DataFrame()

# Add in values from test and train to calculate change from zero, once include predictions - if not merging, order matters for first col bc that will define index

baseline = pd.merge(train['export_period_total'], test['export_period_total'], how='right', left_index=True, right_index=True)
baseline = baseline.rename(index=str, columns={'export_period_total_x':'export_period_train', 'export_period_total_y':'export_period_test'})
baseline

baseline.describe()

#%%
# Calculate pct change
baseline['export_pct_change'] = (baseline['export_period_test']-baseline['export_period_train'])/baseline['export_period_train']
baseline

#%%
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

#%%
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

#%%
# Create metric to measure NEW goods - can also do -1 (decrease), 0 (no change), 1 (increase), 2 (new good)
mask_new = (baseline['export_period_test'] > 0) & (baseline['export_period_train'] == 0)
baseline[mask_new]

mask_decrease = (baseline['export_pct_change'] < 0)
mask_stagnant = (baseline['export_pct_change'] == 0)
mask_increase = (baseline['export_pct_change'] > 0) & (baseline['export_period_train'] != 0)

baseline['export_class'] = np.select([mask_new, mask_decrease, mask_stagnant, mask_increase], [2, -1, 0, 1], default=np.nan)

# baseline['export_class'] = np.where(mask_new, 2, np.nan) # same as above
# baseline['export_class'] = np.where(mask_decrease, -1, baseline['export_class'])
# baseline['export_class'] = np.where(mask_stagnant, 0, baseline['export_class'])
# baseline['export_class'] = np.where(mask_increase, 1, baseline['export_class'])

baseline

#%%
# DATA LEAKAGE - need to create train.export_trend_pct_rank standin for test

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

#%%
# Define df_locations df
df_locations = data_clean.get('/df_locations')
df_locations
df6_classes = data_clean.get('/df6_classes')
df6_classes

#%%
# Show standardized prediction values - all countries, products, years
predictions_raw = predictions
predictions_raw

predictions_raw['knn_predictions'][0:10]

# Hard code std and mean values for loop
test_export_value_std = test['export_value'].std()
test_export_value_mean = test['export_value'].mean()
test_export_value_min = test['export_value'].min()
test_export_value_max = test['export_value'].max()

# De-standardize predictions
predictions = []
for i in range(len(predictions_raw)):
    predictions.append((predictions_raw['knn_predictions'][i] * (test_export_value_max - test_export_value_min)) + test_export_value_min)

predictions[:20]
len(predictions)
type(predictions)
# currently a list of arrays
predictions[0:5]

predictions.shape

#%%
# concatenate into a single array
# predictions = np.concatenate(predictions, axis=0)

# Merge predictions into baseline
baseline['predictions'] = predictions
baseline

#%%
# Calculate pct change
baseline['pred_pct_change'] = (baseline['predictions'] - baseline['export_period_train']) / baseline['export_period_train']
baseline

#%%
# Clean the NaNs in export_pct_change
mask_nan = baseline['pred_pct_change'].isnull() # returns values which ARE NaN
baseline[mask_nan]

baseline['pred_pct_change'] = np.where(mask_nan, baseline['predictions'], baseline['pred_pct_change'])
baseline[mask_nan]
# baseline.loc[baseline['export_pct_change'].isnull(), baseline['export_pct_change']] = baseline['export_period_test'] # same as

# Check for nulls
baseline.isnull().values.any()

#%%
# Clean the inf/-inf values
mask_pos = baseline['pred_pct_change'] != np.inf # returns values which are NOT inf
mask_neg = baseline['pred_pct_change'] != -np.inf # returns values which are NOT -inf

baseline[~mask_pos]
baseline[~mask_neg]

baseline['pred_pct_change'] = np.where(~mask_pos, baseline['predictions'], baseline['pred_pct_change'])
baseline[~mask_pos]

baseline['pred_pct_change'] = np.where(~mask_neg, baseline['predictions'], baseline['pred_pct_change'])
baseline[~mask_neg]

#%%
# Merge train and test countries, products, years
cols = ['location_id', 'product_id', 'year']
train_ref = train[cols]
test_ref = test[cols]

train_ref.shape
test_ref.shape

df_reference = pd.merge(train_ref, test_ref, how = 'right', on = ['location_id', 'product_id', 'year'])
df_reference

#%% Merge df6_classes into reference table
df_reference = pd.merge(df_reference, df6_classes[['index', 'name']], how = 'left', left_on = 'product_id', right_on = 'index')
df_reference = df_reference.drop(columns='index')
df_reference

# ensure that the index dtypes are the same
baseline.index = baseline.index.astype('int64')
df_reference.index = df_reference.index.astype('int64')

#%%
# Merge reference and baseline dfs
df_knn = pd.concat([df_reference, baseline], axis = 1)
df_knn.shape
df_knn
df_knn.describe()

#%%
# Which country to make prediction for
df_locations[df_locations['name_en'] == 'France']

location = int(df_locations[df_locations['name_en'] == 'France']['index'])
location

#%%
# Return recommendations for a given country
df_recs = df_knn.loc[(df_knn['location_id'] == location) & (df_knn['year'] == 2014)]

# Return the nlargest pct change
df_recs.nlargest(50, columns = 'pred_pct_change')

#%%
# Redefine problem to only include change from 0

# Check what these products are
df_knn[df_knn['export_period_train'] == 0].shape

# Redefine the recommendation df
df_recs_new = ( df_knn.loc[(df_knn['location_id'] == location) & (df_knn['year'] == 2014) &
                (df_knn['export_period_train'] == 0)] )

df_recs_new
# Make recs for new products
df_recs_new.nlargest(20, columns = 'pred_pct_change', keep = 'first')

#%%
# Validation metrics
# Calculate cosine_similarity

# just a test
cosine_similarity([[1, 0, -1]], [[-1,-1, 0]])
type([[1,0,-1]])

print(cosine_similarity([df_knn['export_period_test']], [df_knn['predictions']]))
print(cosine_similarity([df_knn['export_pct_change']], [df_knn['pred_pct_change']]))

print(mean_absolute_error([df_knn['export_pct_change']], [df_knn['pred_pct_change']]))
print(mean_squared_error([df_knn['export_pct_change']], [df_knn['pred_pct_change']]))

#%%
# Next step is to rank the pct_change actual vs. predict and do MSE validation
df_knn['export_pct_change_rank'] = df_knn['export_pct_change'].rank(ascending=False, method='min')
df_knn.sort_values(by=['export_pct_change_rank'])

# REALLY useful - shows the top predicted export pct change areas - travel/tourism, computer inputs, etc. dominate
df_knn['pred_pct_change_rank'] = df_knn['pred_pct_change'].rank(ascending=False, method='min')
df_knn.sort_values(by=['pred_pct_change_rank'])


#%%
# Calculate MSE between export_pct and pred_pct ranks
print(cosine_similarity([df_knn['export_pct_change_rank']], [df_knn['pred_pct_change_rank']]))
print(mean_absolute_error([df_knn['export_pct_change_rank']], [df_knn['pred_pct_change_rank']]))
print(mean_squared_error([df_knn['export_pct_change_rank']], [df_knn['pred_pct_change_rank']]))

# See if there's much difference between train and test export values
df_knn['export_period_train_rank'] = df_knn['export_period_train'].rank(ascending=False, method='min')
df_knn['export_period_test_rank'] = df_knn['export_period_test'].rank(ascending=False, method='min')
print(cosine_similarity([df_knn['export_period_train_rank']], [df_knn['export_period_test_rank']]))

#%%
# Save df to hdf5
df_knn.to_hdf('data/processed/data_predictions.h5', key='df_knn')
