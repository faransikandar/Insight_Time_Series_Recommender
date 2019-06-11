'''
Faran Sikandar
Insight AI.SV19B
06/08/2019

Project: Net_Align
Description: Using Representation Learning to Improve Recommender Systems for Economic Diversification

Data: Atlas of Economic Complexity
'''

#%%
# imports
import os
import pandas as pd

#%%
# check getcwd
os.getcwd()
os.chdir('/Users/faransikandar/Documents/Insight_AI/Insight_Net_Align/')
os.getcwd()


# open hdf5 file for reading
hdf = pd.HDFStore('./data/raw/data_17_0.h5', mode='r')

hdf.keys()

#%%
# extract country summaries

df_country = hdf.get('/country')
df_country.head()

#%%
# extract hs product data

df2 = hdf.get('/country_hsproduct2digit_year')
df4 = hdf.get('/country_hsproduct4digit_year')
df6 = hdf.get('/country_hsproduct6digit_year')
df6.shape
df6.head()

# extract country hs product lookbacks

df2_lookback = hdf.get('/country_hsproduct2digit_lookback')
df4_lookback = hdf.get('/country_hsproduct4digit_lookback') # there is no 6digit lookback
df4_lookback.head()

# extract country summaries lookback

df_country_lookback = hdf.get('/country_lookback')
df_country_lookback.head()

#%%

# extract country-


#%%
# extract classes
df_classes = hdf.get('/classifications/hs_product')
type(df_classes)
df_classes.shape
df2_classes = df_classes[df_classes['level'] == '2digit']
df4_classes = df_classes[df_classes['level'] == '4digit']
df6_classes = df_classes[df_classes['level'] == '6digit']

df6_classes.shape
df6_classes.head()



#%%
# close hdf file
hdf.close()

df.head()
