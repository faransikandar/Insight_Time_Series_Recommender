'''
Faran Sikandar
Insight AI.SV19B
06/08/2019

Project: Net_Align
Description: Using Representation Learning to Improve Recommender Systems for Economic Diversification

Data: Atlas of Economic Complexity

Notes:
- Recommender system code inspired from https://github.com/WillKoehrsen/wikipedia-data-science/blob/master/notebooks/Book%20Recommendation%20System.ipynb

'''

#%%
# imports
import os
import pandas as pd
from collections import Counter, OrderedDict
from itertools import chain

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

# define the year for analysis
year = 2017
df6_2016 = df6[df6['year'] == 2016]
df6_2017 = df6[df6['year'] == 2017]
df6_2017.shape
df6_2017.head()

# check not null vals
df6_2016_clean = df6_2016[df6_2016['export_value'].notnull()]
df6_2016_clean.shape
df6_2017_clean = df6_2017[df6_2017['export_value'].notnull()]
df6_2017_clean.shape

# extract country hs product lookbacks
df2_lookback = hdf.get('/country_hsproduct2digit_lookback')
df4_lookback = hdf.get('/country_hsproduct4digit_lookback') # there is no 6digit lookback
df4_lookback.head()

# extract country summaries lookback
df_country_lookback = hdf.get('/country_lookback')
df_country_lookback.tail()

print(df_country.shape)
print(df_country_lookback.shape)

# look at just one set of lookbacks for a given year
df_country_lookback3 = df_country_lookback[df_country_lookback['lookback_year'] == 2014]
df_country_lookback3.shape

#%%
# create counter to find most common countries and products
def count_items(data):
    '''
    input: data, in the form of a list # see if you can modify to take pd df and convert to list
    output: ordered dict of counts of objects in pd_data
    '''
    # create counter object
    counts = Counter(data)

    # sort by highest count first and place in ordered dict
    counts = sorted(counts.items(), key = lambda x: x[1], reverse = True)
    counts = OrderedDict(counts)

    return counts

#%%

# get product counts
list(df6_2016_clean)
unique_products = list(chain(*[list(set(df6_2016_clean['product_id'])) for i in df6_2016_clean]))

product_counts = count_items(unique_products)
list(product_counts.items())[:10]

unique_countries = list(chain(*[list(set(df6_2016_clean['location_id'])) for i in df6_2016_clean]))

country_counts = count_items(unique_countries)
list(country_counts.items())[:10]

#%%


# extract country


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
