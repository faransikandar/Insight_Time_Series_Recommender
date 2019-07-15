# Insight Time Series Recommender

Combining deep learning recommender systems and time series analysis to recommend export growth areas across countries, and products, and time.

## Project Overview

How do investors and central governments determine where to invest in an economy? While time series forecasting is one potential way of predicting growth in a given sector over time, there are not good methods for comparing multiple sectors across multiple countries over time. More, while these forecasts are a reasonable measure of the state of the world "as it is" changing, they are not necessarily economically sound recommendations of how a country "should" change - i.e. based on what is easiest for them, their relative strengths, and similarities to other economies.

In order to solve this problem, I take a recommendations-oriented approach to forecasting rather than a purely predictions-oriented approach.

This project combines collaborative filtering recommender systems and time series analysis in order to provide recommendations for export growth across 250 countries and territories, over 5,000 product areas, and 20 years. The model is trained on data from 1995-2004 and makes recommendations on data from 2005-2014.

The source code includes a command line interface - the user can input a country/territory and the model will recommend the top 50 product/service areas which that country/territory should export, based on measured country-product similarities.

Slides describing this project and be found here: http://bit.ly/time-rec-demo

## Requisites

- Linux or MacOS. Windows may work as well, but I have not tested it.
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

3. Install pip within your conda environment and install the project dependencies from `requirements.txt`
```
conda install --file build/requirements_conda.txt
pip install -r build/requirements_pip.txt
```
* pip should already come with conda, but in case it isn't there you can use `conda install pip` - check installations in current environment with `conda list`
* separate installs are needed for conda and pip because different packages are available from each
* note that order matters here (conda install goes first) because scikit-surprise (installed with pip, used for alternative algorithms) forces a manual install of numpy first (installed with conda)

## Run an Example Script - Clean Data, Train, and Make Recommendations

4. Clean Data, Train, and Make Recommendations in one pipeline - using 2-digit product/service area specificity (the full model is trained on 6-digit specificity). This will take ~5 min.
```
python -m source.time_series_rec_example
```

**Note that you will be prompted for a user input for the country/territory name.*** Spelling/capitalization matters. Also please note that some of the broader regions (e.g. Asia) do not have recommendation data available, although they appear in the options list.

## Results

### Collaborative Filtering Over Time - An Example

Collaborative filtering works well for static time slices. But how might recommendations change over time? We want to be able to account for changes in trends over time (note the arrows signifying an increase in England's trade in technology from 1995 to 2005).

![Collab Filtering Over Time](images/Insight_Collab_Filtering_Over_Time.png)

### Positive Results - Making Recommendations that  Accurately Predict Growth

We get good results. For example, we are able to make recommendations for Ireland that Information Communication Technologies (ICT), Transport, and Financial Services would be its highest growth export sectors. Indeed, *even though its economy was predominantly machinery and chemicals-focused in 1995*, it shifted largely to services including ICT, Transport, and Financial by 2014.

**To be clear, these product/service areas were not part of the training data, nor were they the highest growth areas in Ireland during the training period - yet we still predicted (i.e. recommended) them to be the highest growth areas!**

![Insight Recs Ireland](images/Insight_Recs_Ireland.png)

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
