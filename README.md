# Insight Time Series Recommender

Combining deep learning recommender systems and time series analysis to recommend export growth areas across countries, and products, and time.

## Project Overview

How do investors and central governments determine where to invest in an economy? While time series forecasting is one potential way of predicting growth in a given sector over time, there are not good methods for comparing multiple sectors across multiple countries over time. More, while these forecasts are a reasonable measure of the state of the world "as it is" changing, they are not necessarily economically sound recommendations of how a country "should" change - i.e. based on what is easiest for them, their relative strengths, and similarities to other economies.

In order to solve this problem, I take a recommendations-oriented approach to predicting rather than a purely forecasting-oriented approach.

This project combines collaborative filtering recommender systems and time series analysis in order to provide recommendations for export growth across 250 countries and territories, over 5,000 product areas, and 20 years. The model is trained on data from 1995-2004 and makes recommendations on data from 2005-2014.

The source code includes a command line interface - the user can input a country/territory and the model will recommend the top 50 product/service areas which that country/territory should export, based on measured country-product similarities.

Slides describing this project and be found here: http://bit.ly/time-rec-demo

## Requisites

- `git`
- `conda`
  * Install [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/)
- `pip`
  * For example, on Ubuntu 18.04 run `sudo apt-get install python3-pip`
- `python` (Python version 3.6.8)

## Setup

1. Clone repository.
```
git clone https://github.com/faransikandar/Insight_Time_Series_Recommender.git
```

2. Set up a conda environment and activate it. (Here, the environemnt is called 'time-series-rec', but you can call it whatever you like.)
```
cd Insight_Time_Series_Recommender
conda create --name time-series-rec python=3.6.8
conda activate time-series-rec
```

3. Install the project dependencies from `requirement_conda.txt` and `requirements_pip.txt`.
```
conda install --file build/requirements_conda.txt
pip install -r build/requirements_pip.txt
```
* pip should already come with conda, but in case it isn't there you can use `conda install pip` - check installations in current environment with `conda list`
* separate installs are needed for conda and pip, because different packages are available from each
* note that order matters here (conda install goes first). this is because scikit-surprise (installed with pip, used for alternative algorithms) forces a manual install of numpy first

## Run an Example Script - Clean Data, Train, and Make Recommendations

Clean Data, Train, and Make Recommendations in one pipeline - using 2-digit product/service area specificity (the full model is trained on 6-digit specificity). **This will take ~3-5 min.**

```
python -m source.time_series_rec_example
```

**NOTE ON USER INPUT:** You will be prompted for a user input for the country/territory name near the end of the script. Spelling and capitalization matters. Please note that some of the broader regions (e.g. Asia) do not have recommendation data available, although they appear in the options list.

**NOTE ON DISPLAY OF OUTPUTS:** The output table is a rather wide data table, so it's best to use a full screen view in your terminal window.

## Data Sources

This project was developed using data from the ATLAS of Economic Complexity: http://atlas.cid.harvard.edu/

## Running Custom Models

If you'd like to be able to make recommendations on the full model, download `data_17_0.h5` from the downloads sections of the ATLAS website and place the file in the subdirectory `data/raw`. In the `main()` section of each of the main processing scripts (`data_cleaner.py`, `model_builder.py`, and `rec_predicter.py`), set the following variables:
- `dict_data_key_raw = 'data_full'`
- `dict_data_key_clean = 'data_full_clean'`
- `model_name = 'model_5L_full-04-0.8404.hdf5'` (set name to your liking if you'd like to train from scratch)
- `model_history = 'model_5L_full'` (set name to your liking if you'd like to train from scratch)
- `model_load = True` (set to `False` if training from scratch - you will need a GPU)

Leave the other variables as they are.

If you'd like to be able to do very fasts runs on the data (cleaning, training, predicting), you can train on a 2-country sample of data. In the `main()` section of each of the main processing scripts (`data_cleaner.py`, `model_builder.py`, and `rec_predicter.py`), set the following variables:
- `dict_data_key_raw = 'data_sample'`
- `dict_data_key_clean = 'data_sample_clean'`
- `model_name = 'model_5L_full_sample-10-0.1711.hdf5'` (set name to your liking if you'd like to train from scratch)
- `model_history = 'model_5L_full_sample'` (set name to your liking if you'd like to train from scratch)
- `model_load = True` (set to `False` if training from scratch - you will need a GPU)

**NOTE ON MODEL NAMING:** If you want to train a model from scratch, simply write the base name. The epoch and loss of the best model are checkpointed and added to the file name that is saved to the `models` directory.

## Results

### Collaborative Filtering Over Time - An Example

Here, we outline a two-country, three-product recommendation scenario. Recommendations are made based on country-product (i.e. user-item) similarity. Collaborative filtering recommenders work well for static time slices. But how might recommendations change over time? We want to be able to account for changes in trends over time (note the arrows signifying an increase in England's trade in "technology" from 1995 to 2005).

![Collaborative Filtering Over Time](images/Insight_Collab_Filtering_Over_Time.png)

### Positive Results - Making Recommendations that  Accurately Predict Growth

We get good results! We calculate a Cosine Similarity of 0.854 between the predicted rank of export change percentage and actual rank of export change percentage. This means that we are rather accurately able to predict what the export change percentage (and its rank amongst the total) for a given product/service area will be.

As a more specific example, we are able to make recommendations for Ireland that Information Communication Technologies (ICT), Transport, and Financial Services would be its highest growth export sectors. Indeed, *even though its economy was predominantly machinery and chemicals-focused in 1995*, it shifted largely to services including ICT, Transport, and Financial by 2014.

**To be clear, these product/service areas were not part of the training data, nor were they the highest growth areas in Ireland during the training period - yet we still predicted (i.e. recommended) them to be the highest growth areas!**

### Recommendations for Ireland 2005-2014

![Recommendations Ireland](images/Insight_Recs_Ireland.png)

## Acknowledgements

This project was developed at the Insight Artificial Intelligence Fellowship. I am grateful for the support and guidance the Insight community provided, as well as to Ricardo Hausmann and Cesar Hidalgo for providing the foundational research upon which this project could be built.

## Project Structure
```
├── LICENSE
├── README.md             <- The top-level README for developers using this project
│
├── build
│   └── requirements.txt  <- The requirements file for reproducing the analysis environment
│
├── data
│   ├── example           <- data_example.h5 dataset for running fast examples locally
│   └── preprocessed      <- data_prep_{}.h5 files including intermediate calculations
│   └── processed         <- data_clean_{}.h5 files used in final inference
│   └── raw               <- full, raw data used for training full model
│
├── images                <- Images used in README.md, including sample output
│
├── models                <- Various pre-trained models
│
├── source                        <- Source code, various python executable files
│   ├── data_cleaner.py           <- Cleans raw data
│   └── data_example_creation.py  <- Create example data for fast runs of model testing
│   └── data_loader.py            <- Function for loading data in different modules
│   └── model_builder.py          <- Build (train or load) DL models
│   └── rec_predicter.py          <- Make predictions/recommendations
│
├── requirements.txt      <- The requirements file for reproducing the analysis environment
│
└── README-ECON.md        <- Detailed README outlining the economic theory behind the model
```
