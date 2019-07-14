# Insight Time Series Recommender
This project combines deep learning, collaborative filtering recommender systems with time series features in order to provide recommendations for export growth areas across countries and years over a twenty-year period. This is a promising potential tool for both investors and central governments to determine where to invest their money and resources. In general terms, the model finds countries which have similar economic profiles to each other and makes recommendations for export areas based on which areas each country is inactive or sub-optimally active in (relative to the comparison countries). For example, if the total economy consists of 10 goods Country A and Country B are found to be similar based on their export of goods 1-9, but Country A trades in the 10th good while Country B doesn't, we can recommend that Country B should also move into the 10th good. This idea leverages the user-item collaborative filtering, as used by companies like Amazon or Netflix, which are recommending different products or movies to different users. Here, countries are the "users" and export product/service areas are the "items."

The project leverages data from the ATLAS of Economic Complexity, including 250 countries, 6,000+ product and service areas, and 22 years, comprising 1 million + links (or iterations of country-product trade instances). The model is trained on data from 1995-2004 and tested on data from 2005-2014 - so, the way of thinking about this problem is "how can we use information about the structure of the economy from one slice in time in order to predict the structure of the economy in the following slice of time?" A 10-year period is used for training because shifts in trade patterns occur gradually and the data includes high variance on a strictly year-to-year basis (partially due to reporting errors).

While the actual magnitude of growth can be better predicted by individual growth models (e.g. time series regression or RNNs) for specific product/service areas, it is novel to use recommender systems to generate _hypotheses_ about potential growth areas in economics or finance.


##
WHAT IS THE PROBLEM?
How do investors and central governments determine where to invest in an economy? Countries exhibit significant change year-to-year with respect to what products and services they export, but it's difficult to predict which changes will take place - let alone what economically sound decisions are for sustained growth. While time series forecasting is one potential way of predicting growth in a given sector over time, there are not good methods for comparing multiple sectors across multiple countries over time.

##
WHAT IS THE SOLUTION?
This project combines deep learning, collaborative filtering recommender systems with time series features in order to provide recommendations for export growth areas across countries and years over a twenty-year period. The model is trained on data from 1995-2004 and predicts on data from 2005-2014. Recommendations for a given country are made based on identifying export areas which similar countries export, but the country in question doesn't. According to economic theory, this is a recommendation of what countries "should" export based on their economic profiles. We validate these recommendations against what actually happened.

The project leverages data from 250 countries, 6,000+ product/service areas, and 20 years, resulting in 1 million+ links (i.e. iterations of country-product pairs).

The deliverable includes a command line interface - the user can input a country and the model will recommend the top 50 product/service areas which that country should export, based on country-country similarity.


## Contributions to Deep Learning (and Economics)

This project makes 4 unique contributions to deep learning:

1) Incorporates time-series information into deep learning recommender systems. DEVELOPING TIME SERIES TREND OF EACH TEN-YEAR TRAIN/TEST PERIOD BY COMPARING OVERALL TREND BETWEEN FIRST 5-YEAR CHUNK

2) Develops a framework for combining multiple embeddings and continuous feature classes to further specify the model.

3) Develops a methodology for effectively standardizing economic data with high variance and a large class imbalance in order to be trained on neural networks. SPECIFICALLY, I found that standardizing across all countries and years for the total export value. STANDARDIZING BY COUNTRY AND YEAR FOR THE EXPORT TREND.

4) Introduces a new application area for deep learning and recommender systems more generally - specifically, economic trade and financial networks.

## Motivations from Economics

The idea behind the project was motivated from some existing hypotheses and problems in economics:

1) Economic theory tells us that countries will find it easier to move into product/service areas that are similarly complex and technologically close to their existing product/service areas. As such, countries with similar export profiles (similar products/services with similar complexities, distance to each other, etc.) can likely learn from each other's development pathways and perhaps will follow similar development pathways. These hypotheses suggest that a recommender system may be a useful tool to leverage in order to derive recommendations or hypotheses about what product/service areas a country _should_ move into. The results from such a recommendation can be treated as a prediction of what _will_ - and can be validated against what actually happens in reality. Thus, we validate what we think _should_ happen against what _will_ or _does_ actually happen.

2) Countries with more diversified export profiles exhibit greater and more stable economic growth. This further motivates the need for investors/countries to be able to predict which new areas they could diversify into.

3) Country-level export profiles change dramatically over time, opening up opportunities both for economic growth, though sometimes at slow increments. The reasons that a country does or does not move into a given product/service area may differ, which make it difficult to validate what "should" happen relative to what actually "does" happen. Specifically, there are two scenarios to keep in mind:

  A) Country A moves into a certain product/service area because it think it's a good move (e.g. subsidizing an industry), even though it really isn't. In this case, exports go up, perhaps for a short time, but the change doesn't persist and a model at time 0 (before the country moves into this export area and provided that other countries haven't made a similar blunder) would have no way of predicting this type of "should."

  B) Country A "should" objectively move into a certain export area, according to this user-item recommender system model and economic complexity theory in general, but it isn't unable to do so due to various industrial, labor market, or political constraints, for example. In this case, exports don't go up and, while the model predicts that the country "should" move into this export area, validation on real data fails because - in reality - Country A doesn't move into the new export area.

Given these motivations (and problems), a deep-learning-based recommender system can provide an important avenue for helping solve some outstanding problems in economics and provide direct benefit to both investors and central governments.

# High-Level Findings

1) Trade areas with large growth globally seem to dominate the top 10 recommendations for all countries (e.g. ICT, tourism, oil, etc.) Beyond the top 10, we begin to see more personalization specific to each country.

2) Because the model is searching for similarly structured economies, and any change relative to 0 will be the easiest to detect, the model should be particularly good at making recommendations in areas where a country is lagging relative to its counterparts.

3) The model may do less well at predicting changes between already well-established export areas.


## Further Enhancements

The model is a proof of concept and can be further improved through some of the following approaches:

1) Using a graph alignment score to further tune the model, or entirely re-hashing the problem using graph neural networks. This would require building a separate attributed graph database for every country-year iteration. While cumbersome, this could provide outstandingly better results.

2) Re-framing as a classification problem - e.g. is the product/service area a new good or not?

3) Re-training the model using oversampling of the minority class. i.e. Most products/services are 0 for most countries - this skews the overall predictions downward; so we should oversample any products/services which are actually traded).

3) Individual prediction of specific product/service areas on a time-series or LSTM model

## Data Sources

ATLAS of Economic Complexity: http://atlas.cid.harvard.edu/

## Network Design

![Inspiration from NLP Network Combining Embedding + Continuous Features]
(http://digital-thinking.de/wp-content/uploads/2018/07/combine.png)

## Collaborative Filtering

![Example of Recommender System Algorithms]
(https://cdn-images-1.medium.com/max/1200/1*mz9tzP1LjPBhmiWXeHyQkQ.png)
