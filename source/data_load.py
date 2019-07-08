#%%
import os
import pandas as pd
import sys

#%%
def data_load():

    # load paths
    def load_paths(gdrive = False, ide = False):
        '''
        input: choose whether you are running the script in gdrive, as a shell, or locally (e.g. ide)
        output: relative paths for directories, hdf_filename || can add other directories and files as needed
        '''
        cwd = os.getcwd()
        cwd
        # for shell
        try:
            directory = os.path.dirname(os.path.abspath(__file__))
            hdf_filename = os.path.join(directory,'../data/raw/data_17_0.h5')
            directory = os.path.dirname(os.path.abspath(__file__))
            clean_filename = os.path.join(directory,'../data/processed/data_clean.h5')
        except NameError:
            # for gdrive
            if gdrive == True:
                directory = os.path.dirname(os.path.abspath('data_17_0.h5'))
                hdf_filename = '/content/gdrive/My Drive/Colab Notebooks/Insight_Net_Align/data/raw/data_17_0.h5'
                clean_filename = '/content/gdrive/My Drive/Colab Notebooks/Insight_Net_Align/data/processed/data_clean.h5'
            # for local (e.g. IDE)
            if ide == True:
                directory = os.path.dirname(os.path.abspath('data_17_0.h5'))
                directory
                hdf_filename = os.path.join(directory,'data/raw/data_17_0.h5')
                hdf_filename
                clean_filename = os.path.join(directory,'data/processed/data_clean.h5')
                clean_filename
        dict_paths_def = {'cwd':cwd, 'directory':directory, 'hdf_filename':hdf_filename, 'clean_filename':clean_filename}
        return dict_paths_def

    dict_paths = load_paths(gdrive = False, ide = True)

    # check path names
    dict_paths

    # load the data
    hdf = pd.HDFStore(dict_paths['hdf_filename'], mode = 'r')
    data_clean = pd.HDFStore(dict_paths['clean_filename'], mode = 'r') # mode = 'r'
    dict_load = {'hdf':hdf,'data_clean':data_clean}
    return dict_load

data_load()

loader = data_load()['data_clean']

loader.keys()

temp = loader.get('/train')

temp
