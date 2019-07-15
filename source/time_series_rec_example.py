#%%
# import libraries for master file
import os
import source.data_loader
import source.data_cleaner
import source.model_builder
import source.rec_predicter

#%%
source.data_cleaner.main()

#%%
source.model_builder.main()

#%%
source.rec_predicter.main()
