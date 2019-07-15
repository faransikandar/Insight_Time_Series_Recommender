#%%
# import libraries for master file
import os
import source.data_loader
import source.data_cleaner
import source.model_builder
import source.rec_predicter

#%%
# clean data - call the data_cleaner() main function
source.data_cleaner.main()

#%%
# train or load a model - call the model_builder() main function
source.model_builder.main()

#%%
# make predictions/recommendations - call the rec_predicter main function
source.rec_predicter.main()
