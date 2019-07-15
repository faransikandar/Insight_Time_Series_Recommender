#%%
# import libraries
import os
import numpy as np
import pandas as pd

from source.data_loader import *

#%%
# Recommender system code inspired from https://github.com/WillKoehrsen/wikipedia-data-science/blob/master/notebooks/Book%20Recommendation%20System.ipynb

#%%
# If using Google Colab
'''
# check relevant TF, keras, and GPU connections

# show which version of TF working
!pip show tensorflow

# show which version of keras
!pip show keras

# check GPU connection
import tensorflow as tf
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))
'''

'''
# Setup Google Drive + Get the Data

# mount Google Drive locally
from google.colab import drive
drive.mount('/content/gdrive')
'''

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
# normalize by country and year - this may be redundant since we already made export_pct
def norm_minmax(data,targets):
    return (data[targets]-data[targets].min())/(data[targets].max()-data[targets].min())

def norm_std(data,targets):
    return (data[targets]-data[targets].mean())/(data[targets].std())

#%%
def define_df_clean(dict_data_key_raw):
    '''
    Inputs:
    dict_data_key_raw:  'data_full', i.e. 6digit product categories
                        'data_sample', i.e. 2-country data sampled from 2digit product categories
                        'data_2digit', i.e. 2digit product categories
                        dict_data_key_raw includes raw trade data, locations, and product classes
    ________________________
    Outputs:
    df: dataframe of trade data to process in data_cleaning() - either 6digit (full), 2digit, or sample
    ________________________
    '''
    # load the data using data_load()
    dict_data = data_load()
    dict_data

    hdf = dict_data[dict_data_key_raw]
    hdf
    #data_clean = dict_data['data_clean']

    # preview keys in HDF
    hdf.keys()

    if dict_data_key_raw == 'data_full':
        df = hdf.get('/country_hsproduct6digit_year') # 6 digit

    elif dict_data_key_raw == 'data_sample':
        df = hdf.get('/df2_sample')

    elif dict_data_key_raw == 'data_2digit':
        df = hdf.get('/df2')

    # clean the NaNs
    try:
        df = df.fillna(0)
    except:
        df

    # filter out the negatives
    mask = df['export_value'] < 0
    df = df[~mask]

    return df

#%%
def data_cleaning(df):
    '''
    Inputs:
    df: dataframe defined from above
    ________________________
    Outputs:
    dict_cleaned: dictionary including pandas dfs {'clean':clean,'prep':prep} of cleaned data as well as various preprocessed data steps

    # define the clean files
    clean = ( {'train':train, 'test':test} )

    # define the prep files
    prep = ( {'df':df, 'df_groupsum':df_groupsum, 'df_95_04':df_95_04, 'df_05_14':df_05_14, 'df_95_04_sum1':df_95_04_sum1,
    'df_95_04_sum2':df_95_04_sum2, 'df_95_04_trend':df_95_04_trend, 'df_05_14_sum1':df_05_14_sum1, 'df_05_14_sum2':df_05_14_sum2,
    'df_05_14_trend':df_05_14_trend, 'train_pct_norm':train_pct_norm, 'train_pct_std':train_pct_std, 'train_trend_std':train_trend_std,
    'test_pct_norm':test_pct_norm, 'test_pct_std':test_pct_std, 'test_trend_std':test_trend_std} )
    ________________________
    '''
    # Calculate totals by country and year
    # df6.groupby(['location_id','year']).sum().reset_index() # if don't do reset_index(), then loc and year b/c part of index - does all columns
    print("##################################################")
    print('Calculating grouped sums for export values...')

    # sum the exports/imports, by location and year - will be used for improved normalization by country
    df_groupsum = ( group_sum(df=df,groups=['location_id','year'],targets=['export_value','import_value'])
    .rename(index=str, columns={'export_value':'export_total_loc_year', 'import_value':'import_total_loc_year'}) )

    df_groupsum

    df_groupsum.describe()

    # filter the data for a 10 year TRAIN range
    df_95_04 = data_filter(df=df,filter='year',values=range(1995,2005))
    df_95_04.head(10)

    # filter the data for a 10 year TEST range
    df_05_14 = data_filter(df=df,filter='year',values=range(2005,2015))
    df_05_14.head(10)

    # TRAIN sum the exports/imports across the FIRST half of the time slice - for trend analysis
    df_95_04_sum1 = ( df_95_04.loc[df_95_04['year'].isin(range(1995,2000))].groupby(['location_id','product_id'])['export_value','import_value']
    .sum().reset_index().rename(index=str, columns={'export_value':'export_period1', 'import_value':'import_period1'}) )

    df_95_04_sum1

    # TRAIN sum the exports/imports across the SECOND half of the time slice - for trend analysis
    df_95_04_sum2 = ( df_95_04.loc[df_95_04['year'].isin(range(2000,2005))].groupby(['location_id','product_id'])['export_value','import_value']
    .sum().reset_index().rename(index=str, columns={'export_value':'export_period2', 'import_value':'import_period2'}) )

    df_95_04_sum2

    # TRAIN total exports/imports for a given product over ENTIRE 10 year period
    df_95_04_sum_total = ( df_95_04.loc[df_95_04['year'].isin(range(1995,2005))].groupby(['location_id','product_id'])['export_value','import_value']
    .sum().reset_index().rename(index=str, columns={'export_value':'export_period_total', 'import_value':'import_period_total'}) )

    df_95_04_sum_total

    # TEST sum the exports/imports across the FIRST half of the time slice - for trend analysis
    df_05_14_sum1 = ( df_05_14.loc[df_05_14['year'].isin(range(2005,2010))].groupby(['location_id','product_id'])['export_value','import_value']
    .sum().reset_index().rename(index=str, columns={'export_value':'export_period1', 'import_value':'import_period1'}) )

    df_05_14_sum1

    # TEST sum the exports/imports across the SECOND half of the time slice - for trend analysis
    df_05_14_sum2 = ( df_05_14.loc[df_05_14['year'].isin(range(2010,2015))].groupby(['location_id','product_id'])['export_value','import_value']
    .sum().reset_index().rename(index=str, columns={'export_value':'export_period2', 'import_value':'import_period2'}) )

    df_05_14_sum2

    # TEST total exports/imports for a given product over ENTIRE 10 year period
    df_05_14_sum_total = ( df_05_14.loc[df_05_14['year'].isin(range(2005,2015))].groupby(['location_id','product_id'])['export_value','import_value']
    .sum().reset_index().rename(index=str, columns={'export_value':'export_period_total', 'import_value':'import_period_total'}) )

    df_05_14_sum_total

    print("##################################################")
    print('Calculating time trends for train and test periods...')
    # calculate and merge sum2, period_total and export/import trends back into sum1 df; fill NaNs with 0 (if 0 base value)
    df_95_04_trend = ( df_95_04_sum1.assign( export_period2 = df_95_04_sum2['export_period2'], import_period2 = df_95_04_sum2['import_period2'],
    export_trend = lambda x: ((x.export_period2 - df_95_04_sum1['export_period1'])/x.export_period1).fillna(0),
    import_trend = lambda x: ((x.import_period2 - df_95_04_sum1['import_period1'])/x.import_period1).fillna(0),
    export_period_total = df_95_04_sum_total['export_period_total'], import_period_total = df_95_04_sum_total['import_period_total'] ) )

    df_05_14_trend = ( df_05_14_sum1.assign( export_period2 = df_05_14_sum2['export_period2'], import_period2 = df_05_14_sum2['import_period2'],
    export_trend = lambda x: ((x.export_period2 - df_05_14_sum1['export_period1'])/x.export_period1).fillna(0),
    import_trend = lambda x: ((x.import_period2 - df_05_14_sum1['import_period1'])/x.import_period1).fillna(0),
    export_period_total = df_05_14_sum_total['export_period_total'], import_period_total = df_05_14_sum_total['import_period_total'] ) )

    # how to use assign to create multiple values in df
    # df = df.assign(Val10_minus_Val1 = df['Val10'] - df['Val1'], log_result = lambda x: np.log(x.Val10_minus_Val1) )

    df_95_04_trend
    df_05_14_trend

    # impute export inf/-inf with max/min trend for 95_04
    mask_pos = df_95_04_trend['export_trend'] != np.inf
    mask_pos
    mask_neg = df_95_04_trend['export_trend'] != -np.inf
    mask_neg
    df_95_04_trend[~mask_pos]

    #df_95_04_trend.loc[~mask_pos, 'export_trend'] = df_95_04_trend.loc[mask_pos, 'export_trend'].max() # old method
    #df_95_04_trend.loc[~mask_neg, 'export_trend'] = df_95_04_trend.loc[mask_neg, 'export_trend'].min()
    df_95_04_trend['export_trend'] = np.where(~mask_pos, df_95_04_trend['export_period2'], df_95_04_trend['export_trend']) # if div by 0, replaces inf w/ export_period2 value
    df_95_04_trend['export_trend'] = np.where(~mask_neg, -df_95_04_trend['export_period2'], df_95_04_trend['export_trend']) # if div by 0, replaces -inf w/ -export_period2 value

    # impute export inf/-inf with max/min trend for 05_14
    mask_pos = df_05_14_trend['export_trend'] != np.inf
    mask_pos
    mask_neg = df_05_14_trend['export_trend'] != -np.inf
    mask_neg
    df_05_14_trend[~mask_pos]

    # df_05_14_trend.loc[~mask_pos, 'export_trend'] = df_05_14_trend.loc[mask_pos, 'export_trend'].max()
    # df_05_14_trend.loc[~mask_neg, 'export_trend'] = df_05_14_trend.loc[mask_neg, 'export_trend'].min()
    df_05_14_trend['export_trend'] = np.where(~mask_pos, df_05_14_trend['export_period2'], df_05_14_trend['export_trend']) # if div by 0, replaces inf w/ export_period2 value
    df_05_14_trend['export_trend'] = np.where(~mask_neg, -df_05_14_trend['export_period2'], df_05_14_trend['export_trend']) # if div by 0, replaces -inf w/ -export_period2 value

    df_05_14_trend[~mask_pos]

    # impute import inf/-inf with max/min trend for 95_04
    mask_pos = df_95_04_trend['import_trend'] != np.inf
    mask_pos
    mask_neg = df_95_04_trend['import_trend'] != -np.inf
    mask_neg

    df_95_04_trend[~mask_neg]

    # df_95_04_trend.loc[~mask_pos, 'import_trend'] = df_95_04_trend.loc[mask_pos, 'import_trend'].max()
    # df_95_04_trend.loc[~mask_neg, 'import_trend'] = df_95_04_trend.loc[mask_neg, 'import_trend'].min()
    df_95_04_trend['import_trend'] = np.where(~mask_pos, df_95_04_trend['import_period2'], df_95_04_trend['import_trend']) # if div by 0, replaces inf w/ export_period2 value
    df_95_04_trend['import_trend'] = np.where(~mask_neg, -df_95_04_trend['import_period2'], df_95_04_trend['import_trend']) # if div by 0, replaces -inf w/ -export_period2 value

    df_95_04_trend[~mask_pos]

    # impute import inf/-inf with max/min trend for 05_14
    mask_pos = df_05_14_trend['import_trend'] != np.inf
    mask_pos
    mask_neg = df_05_14_trend['import_trend'] != -np.inf
    mask_neg
    df_05_14_trend[~mask_neg]

    # df_05_14_trend.loc[~mask_pos, 'import_trend'] = df_05_14_trend.loc[mask_pos, 'import_trend'].max()
    # df_05_14_trend.loc[~mask_neg, 'import_trend'] = df_05_14_trend.loc[mask_neg, 'import_trend'].min()
    df_05_14_trend['import_trend'] = np.where(~mask_pos, df_05_14_trend['import_period2'], df_05_14_trend['import_trend']) # if div by 0, replaces inf w/ export_period2 value
    df_05_14_trend['import_trend'] = np.where(~mask_neg, -df_05_14_trend['import_period2'], df_05_14_trend['import_trend']) # if div by 0, replaces -inf w/ -export_period2 value

    df_95_04_trend.describe()
    df_95_04_trend

    # merge df_95_04_trend back into d56_95_04 by location and product (will be repeats of summed values)
    train = pd.merge(df_95_04, df_groupsum, on=['location_id','year'], how='inner')
    train = pd.merge(train, df_95_04_trend, on=['location_id','product_id'], how='inner')
    train

    # merge df_05_14_trend back into d56_05_14 by location and product (will be repeats of summed values)
    test = pd.merge(df_05_14, df_groupsum, on=['location_id','year'], how='inner')
    test = pd.merge(test, df_05_14_trend, on=['location_id','product_id'], how='inner')
    test

    # Define train and test - make sure to normalize AFTER this so as not to have data leakage
    cols = ['location_id','product_id','year','export_value','export_total_loc_year','export_period1','export_period2','export_period_total','export_trend']
    train = train.copy(deep=True)
    train = train[cols]
    train

    test = test[cols]
    test = test.copy(deep=True)
    test

    # Calculate product percent of total exports for that country and year
    train['export_pct'] = (train['export_value']/train['export_total_loc_year'])
    train.head()

    test['export_pct'] = (test['export_value']/test['export_total_loc_year'])
    test.head()

    print("##################################################")
    print('Calculating various normalizations/standardizations (this may take several minutes on large datasets)...')
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

    # perform various normalization strategies, grouping by country/year/across all/etc and for raw values vs percents of total
    # normalize by country and year ??? doesn't seem to get me what I want - possible that you don't WANT to normalize by country and year, because perhaps overall global trade of goods is more important
    # takes some processing time - may be a more efficient way to do this

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
    # df_.groupby(['location_id','year']).apply( lambda x: (x['export_pct']-x['export_pct'].min())/(x['export_pct'].max()-x['export_pct'].min()) )
    # do product_id as well - otherwise indices lost?
    # df_train.groupby(['location_id','year','product_id']).apply(norm_minmax, targets='export_pct').to_frame().reset_index()

    # df_train.groupby(['location_id','year'])( (df['export_pct']-df['export_pct'].min())/(df_train['export_pct'].max()-df_train['export_pct'].min()) )
    # df2_2007_norm['export_value'] = (df2_2007['export_value']-df2_2007['export_value'].min())/(df2_2007['export_value'].max()-df2_2007['export_value'].min())
    # df.groupby(['location_id','year'])['export_value'].sum().reset_index()
    '''

    train_pct_std
    test_trend_norm.describe()

    # merge the pct and trend norms in
    train_temp = train.join([train_val_norm['export_val_norm'], train_val_std['export_val_std'], train_pct_norm['export_pct_norm'], train_pct_std['export_pct_std'], train_trend_norm['export_trend_norm'], train_trend_std['export_trend_std']])
    #train_temp = pd.merge(train, train_pct_norm['export_pct_norm'], train_pct_std['export_pct_std'], train_trend_norm['export_trend_norm'], left_index=True, right_index=True)
    train = train_temp
    train

    test_temp = test.join([test_val_norm['export_val_norm'], test_val_std['export_val_std'], test_pct_norm['export_pct_norm'], test_pct_std['export_pct_std'], test_trend_norm['export_trend_norm'], test_trend_std['export_trend_std']])
    #test_temp = pd.merge(test, test_pct_norm['export_pct_norm'], train_pct_std['export_pct_std'], test_trend_norm['export_trend_norm'], left_index=True, right_index=True)
    test = test_temp
    test

    #df_train['export_pct_norm'] = (df_train['export_pct']-df_train['export_pct'].min())/(df_train['export_pct'].max()-df_train['export_pct'].min())

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

    # Final NaN cleaning
    train[train.isnull()]
    mask = train.isnull() == True
    train[mask]
    try:
        train = train.fillna(0)
    except:
        train
    train.isnull().values.any()

    try:
        test = test.fillna(0)
    except:
        test

    test.isnull().values.any()

    # save the clean and prep files in a dict

    # define the clean files
    clean = ( {'train':train, 'test':test} )

    # define the prep files
    prep = ( {'df':df, 'df_groupsum':df_groupsum, 'df_95_04':df_95_04, 'df_05_14':df_05_14, 'df_95_04_sum1':df_95_04_sum1,
    'df_95_04_sum2':df_95_04_sum2, 'df_95_04_trend':df_95_04_trend, 'df_05_14_sum1':df_05_14_sum1, 'df_05_14_sum2':df_05_14_sum2,
    'df_05_14_trend':df_05_14_trend, 'train_pct_norm':train_pct_norm, 'train_pct_std':train_pct_std, 'train_trend_std':train_trend_std,
    'test_pct_norm':test_pct_norm, 'test_pct_std':test_pct_std, 'test_trend_std':test_trend_std} )

    dict_cleaned = {'clean':clean, 'prep':prep}

    return dict_cleaned

def main():
    dict_data_key_raw = 'data_full'
    df = define_df_clean(dict_data_key_raw)

    print("##################################################")
    print('This may take a minute. The data is being cleaned for: ', dict_data_key_raw)

    data_cleaned = data_cleaning(df)
    clean_dict = data_cleaned['clean']
    prep_dict = data_cleaned['prep']

    if dict_data_key_raw == 'data_full':
        file_out_clean = 'processed/data_full_clean.h5'
        file_out_prep = 'preprocessed/data_full_prep.h5'
    elif dict_data_key_raw == 'data_sample':
        file_out_clean = 'processed/data_sample_clean.h5'
        file_out_prep = 'preprocessed/data_sample_prep.h5'
    elif dict_data_key_raw == 'data_2digit':
        file_out_clean = 'processed/data_2digit_clean.h5'
        file_out_prep = 'preprocessed/data_2digit_prep.h5'

    ## Export data to HDF5
    # always make train the first item in the dict for organization's sake + necessary when first creating file to name the first key
    directory = os.path.dirname(os.path.abspath('abcdef')) # essentially os.getcwd()
    directory
    clean_filename = os.path.join(directory,'data',file_out_clean)
    clean_filename
    prep_filename = os.path.join(directory,'data',file_out_prep)
    prep_filename

    print("##################################################")
    print('The train dataframe shape is: ', clean_dict['train'].shape)
    print('The test dataframe shape is: ', clean_dict['test'].shape)

    print("##################################################")
    print('Saving clean and preprocessed files...')

    # can turn the save statements into a def with target save file paths and T/F for example

    if os.path.exists(clean_filename): # error handling for data_clean, which for some reason always shows as open for read only if it already exists (not sure on what's going on here)
        os.remove(clean_filename)

    for k, v in clean_dict.items():
        try:
            if k == 'train':
                v.to_hdf(clean_filename, key=k)
            else:
                v.to_hdf(clean_filename, key=k)
        except NotImplementedError:
            if k == 'train':
                v.to_hdf(clean_filename, key=k, format='t') # categorical variables need to be placed in table format
            else:
                v.to_hdf(clean_filename, key=k, format='t')

    if os.path.exists(prep_filename):
        os.remove(prep_filename)

    for k, v in prep_dict.items():
        try:
            if k == 'df':
                v.to_hdf(prep_filename, key=k, mode='w') # the first file has to be written to be created
            else:
                v.to_hdf(prep_filename, key=k)
        except NotImplementedError:
            if k == 'df':
                v.to_hdf(prep_filename, key=k, mode='w', format='t') # categorical variables need to be placed in table format
            else:
                v.to_hdf(prep_filename, key=k, format='t')

    print('Success!')
    print("##################################################")

if __name__ == "__main__":
    main()
