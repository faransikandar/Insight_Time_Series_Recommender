#%%
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

#%%
# Load the data
dict_data = data_load()
dict_data

data_clean = dict_data['data_clean']
data_clean.keys()
test = data_clean.get('/test')
train = data_clean.get('train')

test.head()

#%%
# Create baseline df for comparison of pct change
baseline = pd.DataFrame()

# Add in values from test and train to calculate change from zero, once include predictions
baseline['export_total_test'] = test['export_total']
baseline['export_total_train'] = train['export_total']
baseline.describe()

#%%
# Calculate pct change
baseline['export_pct_change'] = (baseline['export_total_test']-baseline['export_total_train'])/baseline['export_total_train']
baseline

#%%
# Clean the NaNs in export_pct_change
mask_nan = baseline['export_pct_change'].isnull() # returns values which ARE NaN
baseline[mask_nan]

baseline['export_pct_change'] = np.where(mask_nan, baseline['export_total_test'], baseline['export_pct_change'])
baseline[mask_nan]
# baseline.loc[baseline['export_pct_change'].isnull(), baseline['export_pct_change']] = baseline['export_total_test'] # same as

# Clean the NaNs in export_total in export_total_train
mask_nan = baseline['export_total_train'].isnull()
baseline[mask_nan]

baseline['export_total_train'] = np.where(mask_nan, 0, baseline['export_total_train'])
baseline[mask_nan]

# Check for nulls
baseline.isnull().values.any()

#%%
# Clean the inf/-inf values
mask_pos = baseline['export_pct_change'] != np.inf # returns values which are NOT inf
mask_neg = baseline['export_pct_change'] != -np.inf # returns values which are NOT -inf

baseline[~mask_pos]
baseline[~mask_neg]

baseline['export_pct_change'] = np.where(~mask_pos, baseline['export_total_test'], baseline['export_pct_change'])
baseline[~mask_pos]

baseline['export_pct_change'] = np.where(~mask_neg, -baseline['export_total_test'], baseline['export_pct_change'])
baseline[~mask_neg]

baseline

#%%
# Create metric to measure NEW goods - can also do -1 (decrease), 0 (no change), 1 (increase), 2 (new good)
mask_new = (baseline['export_total_test'] > 0) & (baseline['export_total_train'] == 0)
baseline[mask_new]

mask_decrease = (baseline['export_pct_change'] < 0)
mask_stagnant = (baseline['export_pct_change'] == 0)
mask_increase = (baseline['export_pct_change'] > 0) & (baseline['export_total_train'] != 0)

baseline['export_class'] = np.select([mask_new, mask_decrease, mask_stagnant, mask_increase], [2, -1, 0, 1], default=np.nan)

# baseline['export_class'] = np.where(mask_new, 2, np.nan) # same as above
# baseline['export_class'] = np.where(mask_decrease, -1, baseline['export_class'])
# baseline['export_class'] = np.where(mask_stagnant, 0, baseline['export_class'])
# baseline['export_class'] = np.where(mask_increase, 1, baseline['export_class'])

baseline

#%%
# Define the model filepath
os.getcwd()

directory = os.path.dirname(os.path.abspath('model_nn_trend_rank-03-0.6580.hdf5'))
directory
model_filename = os.path.join(directory,'models/model_nn_trend_rank-03-0.6580.hdf5')
model_filename
history_filename = os.path.join(directory,'models/history_nn_trend_rank')
history_filename

#%%
# Load the model
model_nn = load_model(model_filename)
model_nn

# Load the history
history_nn = pickle.load(open(history_filename, 'rb'))
history_nn

#%%
# Plot the history of the model
plt.plot(history_nn['loss'])
plt.xlabel('Number of Epochs')
plt.ylabel('Training Error')

#%%
# Evaluate model_nn on 2008 - can set = and add callbacks in order to get history?
print(model_nn.metrics_names)
model_nn.evaluate([test.location_id, test.product_id, test.export_trend_pct_rank], test.export_val_std_all) # takes 21 hours?? on CPU

#%%
# Define df_locations df
df_locations = data_clean.get('/df_locations')
df_locations
df6_classes = data_clean.get('/df6_classes')
df6_classes

#%%
# Show standardized prediction values - all countries, products, years
predictions_raw = model_nn.predict([test.location_id, test.product_id, test.export_trend_pct_rank]) #year, trend for full model
predictions_raw

# Hard code std and mean values for loop
test_export_value_std = test['export_value'].std()
test_export_value_mean = test['export_value'].mean()

# De-standardize predictions
predictions = []
for i in range(len(predictions_raw)):
    predictions.append(predictions_raw[i]*test_export_value_std+test_export_value_mean)

predictions[:20]
len(predictions)
type(predictions)
# currently a list of arrays
predictions[0:2]

# concatenate into a single array
predictions = np.concatenate(predictions, axis=0)

# Merge predictions into baseline
baseline['predictions'] = predictions
baseline

# Calculate pct change
baseline['pred_pct_change'] = (baseline['predictions']-baseline['export_total_train'])/baseline['export_total_train']
baseline

#%%
# Clean the NaNs in export_pct_change
mask_nan = baseline['pred_pct_change'].isnull() # returns values which ARE NaN
baseline[mask_nan]

baseline['pred_pct_change'] = np.where(mask_nan, baseline['predictions'], baseline['pred_pct_change'])
baseline[mask_nan]
# baseline.loc[baseline['export_pct_change'].isnull(), baseline['export_pct_change']] = baseline['export_total_test'] # same as

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

#%%
# Merge reference and baseline dfs
df_final_nn_trend_rank = pd.concat([df_reference, baseline], axis = 1)
df_final_nn_trend_rank.shape
df_final_nn_trend_rank.head()

#%%
# Save df_final to hdf5
df_final_nn_trend_rank.to_hdf('data/processed/data_predictions.h5', key='df_final_nn_trend_rank')

#%%
# Which country to make prediction for
df_locations[df_locations['name_en'] == 'Pakistan']

location = int(df_locations[df_locations['name_en'] == 'Bangladesh']['index'])
location

#%%
# Return recommendations for a given country
df_recs = df_final_nn_trend_rank.loc[(df_final_nn_trend_rank['location_id'] == location) & (df_final_nn_trend_rank['year'] == 2014)]

# Return the nlargest pct change
df_recs.nlargest(50, columns = 'pred_pct_change')

#%%
# Redefine problem to only include change from 0

# Check what these products are
df_final_nn_trend_rank[df_final_nn_trend_rank['export_total_train'] == 0].shape

# Redefine the recommendation df
df_recs_new = ( df_final_nn_trend_rank.loc[(df_final_nn_trend_rank['location_id'] == location) & (df_final_nn_trend_rank['year'] == 2014) &
                (df_final_nn_trend_rank['export_total_train'] == 0)] )

df_recs_new
# Make recs for new products
df_recs_new.nlargest(20, columns = 'pred_pct_change', keep = 'first')

#%%
# Validation metrics
# Calculate cosine_similarity

cosine_similarity([[1, 0, -1]], [[-1,-1, 0]])
type([[1,0,-1]])

print(cosine_similarity([df_final_nn_trend_rank['export_total_test']], [df_final_nn_trend_rank['predictions']]))
print(cosine_similarity([df_final_nn_trend_rank['export_pct_change']], [df_final_nn_trend_rank['pred_pct_change']]))

print(mean_absolute_error([df_final_nn_trend_rank['export_pct_change']], [df_final_nn_trend_rank['pred_pct_change']]))

#%%
