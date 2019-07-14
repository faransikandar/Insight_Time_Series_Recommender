#%%
import os
import pandas as pd
import sys

#%%
# load paths
def load_paths(gdrive = False, ide = False):
    '''
    Inputs:
    gdrive: boolean for whether running scripts in google colab or not
    ide: boolean for wehtehr running scripts in shell or in an ide (where directory paths often default to home directory for project)

    ***Note: sample and 2digit draw from the same initial data_example.h5 raw file, but are processed into separate clean and prep files
    ________________________
    Outputs:    Relative paths for directories, hdf_filename
                Can add other directories and files as needed (e.g. if want to add datasets to train on without overwriting existing data processing)
    ________________________
    '''
    cwd = os.getcwd()
    cwd

    # Define the filenames here - SEE NOTE ABOVE IN DOC STRING
    full_filename = 'data_17_0.h5'
    full_clean_filename = 'data_full_clean.h5'
    sample_filename = 'data_example.h5'
    sample_clean_filename = 'data_sample_clean.h5'
    _2digit_filename = 'data_example.h5'
    _2digit_clean_filename = 'data_2digit_clean.h5'

    try:
        directory = os.path.dirname(os.path.abspath(__file__))
        data_full_filename = os.path.join(directory,'../data/raw/',full_filename)
        data_full_clean_filename = os.path.join(directory,'../data/processed/',full_clean_filename)
        data_sample_filename = os.path.join(directory,'../data/example/',sample_filename)
        data_sample_clean_filename = os.path.join(directory,'../data/example/',sample_clean_filename)
        data_2digit_filename = os.path.join(directory,'../data/example/',_2digit_filename)
        data_2digit_clean_filename = os.path.join(directory,'../data/example/',_2digit_clean_filename)
    except NameError:
        # for gdrive
        if gdrive == True:
            directory = os.path.dirname(os.path.abspath(full_filename))
            data_full_filename = '/content/gdrive/My Drive/Colab Notebooks/Insight_Net_Align/data/raw/'+full_filename
            data_full_clean_filename = '/content/gdrive/My Drive/Colab Notebooks/Insight_Net_Align/data/processed/'+full_clean_filename
            data_sample_filename = 'content/gdrive/My Drive/Colab Notebooks/Insight_Net_Align/data/example/'+sample_filename
            data_sample_clean_filename = 'content/gdrive/My Drive/Colab Notebooks/Insight_Net_Align/data/example/'+sample_clean_filename
            data_2digit_filename = 'content/gdrive/My Drive/Colab Notebooks/Insight_Net_Align/data/example/'+_2digit_filename
            data_2digit_clean_filename = 'content/gdrive/My Drive/Colab Notebooks/Insight_Net_Align/data/example/'+_2digit_clean_filename
        # for local (e.g. IDE)
        if ide == True:
            directory = os.path.dirname(os.path.abspath(full_filename))
            directory
            data_full_filename = os.path.join(directory,'data/raw/',full_filename)
            data_full_filename
            data_full_clean_filename = os.path.join(directory,'data/processed/',full_clean_filename)
            data_full_clean_filename
            data_sample_filename = os.path.join(directory,'data/example/',sample_filename)
            data_sample_filename
            data_sample_clean_filename = os.path.join(directory,'data/example/',sample_clean_filename)
            data_sample_clean_filename
            data_2digit_filename = os.path.join(directory,'data/example/',_2digit_filename)
            data_2digit_filename
            data_2digit_clean_filename = os.path.join(directory,'data/example/',_2digit_clean_filename)
            data_2digit_clean_filename
    dict_paths_def = ( {'cwd':cwd, 'directory':directory, 'data_full_filename':data_full_filename, 'data_full_clean_filename':data_full_clean_filename,
        'data_sample_filename':data_sample_filename, 'data_sample_clean_filename':data_sample_clean_filename,
        'data_2digit_filename':data_2digit_filename, 'data_2digit_clean_filename':data_2digit_clean_filename} )
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
        data_sample = pd.HDFStore(dict_paths['data_sample_filename'], mode = 'r')
    except:
        data_sample = pd.DataFrame()
    try:
        data_sample_clean = pd.HDFStore(dict_paths['data_sample_clean_filename'], mode = 'r')
    except:
        data_sample_clean = pd.DataFrame()
    try:
        data_2digit = pd.HDFStore(dict_paths['data_2digit_filename'], mode = 'r')
    except:
        data_2digit = pd.DataFrame()
    try:
        data_2digit_clean = pd.HDFStore(dict_paths['data_2digit_clean_filename'], mode = 'r')
    except:
        data_2digit_clean = pd.DataFrame()

    dict_load = ( {'data_full':data_full,'data_full_clean':data_full_clean,'data_sample':data_sample,'data_sample_clean':data_sample_clean,
                    'data_2digit':data_2digit, 'data_2digit_clean':data_2digit_clean} )
    return dict_load

#%%
# check the data_load function works
def check_func():
    data_load()

    loader = data_load()['data_full_clean']
    loader.keys()

    temp = loader.get('/train')
    temp = data_load()['data_sample']
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
    data_load()
    print("##################################################")
    print('Success!')
    print("These are the keys in the data_load() dictionary:")
    print(data_load().keys())
    print("##################################################")

if __name__ == "__main__":
    main()
