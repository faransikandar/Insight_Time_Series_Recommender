#%%
import os
import numpy as np
import pandas as pd

from source.data_loader import *

# tmp = pd.HDFStore('data/processed/data_predictions.h5') # mode = 'r' mode = 'a'?
# tmp.keys()
# tmp.close()

#%%
data_load()

data_load().keys()

hdf = data_load()['data_full']
hdf.keys()

df2 = hdf.get('/country_hsproduct2digit_year')
cols = ['location_id', 'product_id', 'year']
# calculation problems if don't convert df2 values to integer (currently are categorical)
for i in cols:
    df2[i] = df2[i].astype(int)

df2_sample = df2[df2['location_id'].isin(range(0,2))]

df_classes = hdf.get('/classifications/hs_product')
df2_classes = df_classes[df_classes['level'] == '2digit']

df_locations = hdf.get('/classifications/location')
df_locations.head()

#%%
# naming example keys as general values and not specific (eg df2) so can be compatible with other fxns in main code
example = ( {'df2':df2, 'df2_sample':df2_sample,'df_classes':df2_classes, 'df_locations':df_locations} )

os.getcwd()

if os.path.exists('data/example/data_example.h5'): # error handling for data_clean, which for some reason always shows as open for read only if it already exists (not sure on what's going on here)
    os.remove('data/example/data_example.h5')

for k, v in example.items():
    try:
        if k == 'df2':
            v.to_hdf('data/example/data_example.h5', key=k, mode='w') # the first file has to be written to be created
        else:
            v.to_hdf('data/example/data_example.h5', key=k)
    except NotImplementedError:
        if k == 'df2':
            v.to_hdf('data/example/data_example.h5', key=k, mode='w', format='t') # categorical variables need to be placed in table format
        else:
            v.to_hdf('data/example/data_example.h5', key=k, format='t')
