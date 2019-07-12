#%%
import os
import pandas as pd
import sys

#%%
# load paths
def load_paths(gdrive = False, ide = False):
    '''
    input: choose whether you are running the script in gdrive, as a shell, or locally (e.g. ide)
    output: relative paths for directories, hdf_filename || can add other directories and files as needed
    '''
    cwd = os.getcwd()
    cwd

    # for shell
    full_filename = 'data_17_0.h5'
    full_clean_filename = 'data_full_clean'
    example_filename = 'data_example.h5'
    example_clean_filename = 'data_example_clean.h5'

    try:
        directory = os.path.dirname(os.path.abspath(__file__))
        data_full_filename = os.path.join(directory,'../data/raw/',full_filename)
        data_full_clean_filename = os.path.join(directory,'../data/processed/',full_clean_filename)
        data_example_filename = os.path.join(directory,'../data/example/',example_filename)
        data_example_clean_filename = os.path.join(directory,'../data/example/',example_clean_filename)
    except NameError:
        # for gdrive
        if gdrive == True:
            directory = os.path.dirname(os.path.abspath(full_filename))
            data_full_filename = '/content/gdrive/My Drive/Colab Notebooks/Insight_Net_Align/data/raw/'+full_filename
            data_full_clean_filename = '/content/gdrive/My Drive/Colab Notebooks/Insight_Net_Align/data/processed/'+full_clean_filename
            data_example_filename = 'content/gdrive/My Drive/Colab Notebooks/Insight_Net_Align/data/example/'+example_filename
            data_example_clean_filename = 'content/gdrive/My Drive/Colab Notebooks/Insight_Net_Align/data/example/'+example_clean_filename
        # for local (e.g. IDE)
        if ide == True:
            directory = os.path.dirname(os.path.abspath(full_filename))
            directory
            data_full_filename = os.path.join(directory,'data/raw/',full_filename)
            data_full_filename
            data_full_clean_filename = os.path.join(directory,'data/processed/',full_clean_filename)
            data_full_clean_filename
            data_example_filename = os.path.join(directory,'data/example/',example_filename)
            data_example_filename
            data_example_clean_filename = os.path.join(directory,'data/example/',example_clean_filename)
            data_example_clean_filename
    dict_paths_def = ( {'cwd':cwd, 'directory':directory, 'data_full_filename':data_full_filename, 'data_full_clean_filename':data_full_clean_filename,
        'data_example_filename':data_example_filename, 'data_example_clean_filename':data_example_clean_filename} )
    return dict_paths_def

#%%
def data_load():
    '''
    inputs: none
    output: dictionary of raw data file (data_17_0.h5, as hdf), clean data file (data_clean.h5, as data_clean), and example data files (data_example.csv, hsproduct2digit.csv)
    '''
    # load dict paths
    dict_paths = load_paths(gdrive = False, ide = True)

    # check path names
    dict_paths

    # load the data - try/except clauses in case these files do not exist on the machine running the program
    try:
        data_full = pd.HDFStore(dict_paths['data_full_filename'], mode = 'r')
    except:
        data_full = pd.DataFrame()
    try:
        data_full_clean = pd.HDFStore(dict_paths['data_full_clean_filename'], mode = 'r') # mode = 'r' mode = 'a'?
    except:
        data_full_clean = pd.DataFrame()
    try:
        data_example = pd.HDFStore(dict_paths['data_example_filename'], mode = 'r')
    except:
        data_example = pd.DataFrame()
    try:
        data_example_clean = pd.HDFStore(dict_paths['data_example_clean_filename'], mode = 'r')
    except:
        data_example_clean = pd.DataFrame()
    dict_load = {'data_full':data_full,'data_full_clean':data_full_clean,'data_example':data_example,'data_example_clean':data_example_clean}
    return dict_load

#%%
# check the data_load function works
def check_func():
    data_load()

    loader = data_load()['data_full_clean']
    loader.keys()

    temp = loader.get('/train')
    temp = data_load()['data_example']
    return temp

# check_func().head()

#%%
# main func
def main():
    '''
    inputs: none
    output: data dict from load data (consisting of raw data hdf5 file and data_clean for both full and example datasets)
    '''
    print("##################################################")
    print('Loading data...')
    print("##################################################")
    data_load()
    print("##################################################")
    print('Success!')
    print("These are the keys in the table:")
    print(data_load().keys())
    print("##################################################")

if __name__ == "__main__":
    main()
