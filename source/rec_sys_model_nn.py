#%%
import os
import sys

from source.data_load import *

#%%
# Load the data
dict_data = data_load()
dict_data

data_clean = dict_data['data_clean']
data_clean.keys()
test = data_clean.get('/test')
train = data_clean.get('train')

#%%
# Define model inputs
n_countries = len(train['location_id'].unique())
n_products = train['product_id'].max()
n_years = train['year'].max()
n_years = 2017
n_trends = len(train['export_trend_norm'].unique()) #int(round(train['export_trend_std'].max()))
#n_trends = 10
n_latent_factors = 10

print(n_countries)
print(n_products)
print(n_years)
print(n_trends)

#%%
# If want to simplify to one year analysis

train_1995 = train[train['year'] == 1995].reset_index()
test_2005 = test[test['year'] == 2005].reset_index()

train_1995 = train_1995.drop(columns=['index'],axis=1)
test_2005 = test_2005.drop(columns=['index'],axis=1)

# Simplify to non-zero inputs
mask = train['export_value'] <= 0
mask_1995 = train_1995['export_value'] <= 0

# Sparse dfs
train_sparse = train[~mask]
train_1995_sparse = train_1995[~mask]
train_sparse.head()

#%%
# Create embeddings

# Creating product embedding path
product_input = Input(shape=[1], name='Product-Input')
product_embedding = Embedding(n_products+1, n_latent_factors, name='Product-Embedding')(product_input)
product_vec = Flatten(name='Flatten-Products')(product_embedding)

# Creating country embedding path
country_input = Input(shape=[1], name='Country-Input')
country_embedding = Embedding(n_countries+1, n_latent_factors, name='Country-Embedding')(country_input)
country_vec = Flatten(name='Flatten-Countries')(country_embedding)

# Creating year embedding path
year_input = Input(shape=[1], name='Year-Input')
year_embedding = Embedding(n_years+1, n_latent_factors, name='Year-Embedding')(year_input)
year_vec = Flatten(name='Flatten-Years')(year_embedding)

# Creating trend embedding path
trend_input = Input(shape=[1], name='Trend-Input')
trend_embedding = Embedding(n_trends+1, n_latent_factors, name='Trend-Embedding')(trend_input)
trend_vec = Flatten(name='Flatten-Trends')(trend_embedding)

#%%
# Define alternative loss metrics

# From: https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html
# Typically used for regression. It’s less sensitive to outliers than the MSE as it treats error as square only inside an interval.

def Huber(yHat, y, delta=1.):
    return np.where(np.abs(y-yHat) < delta,.5*(y-yHat)**2 , delta*(np.abs(y-yHat)-0.5*delta))

#%%
# Full neural network (NN) model - including multiple layers, embeddings, and engineered features
# Compile NN model

# Inspired from: http://digital-thinking.de/deep-learning-combining-numerical-and-text-features-in-deep-neural-networks/

trend_rank_input = Input(shape=[1], name='Trend-Rank-Input')
trend_class_input = Input(shape=[1], name='Trend-Class-Input')

# Concatenate categorical features
conc_cat = Concatenate()([product_vec, country_vec])

# Add fully-connected layers
fc1 = Dropout(0.3)(conc_cat)
fc2 = Dense(128, activation='relu', kernel_initializer='glorot_uniform', kernel_constraint=maxnorm(3))(fc1)
fc3 = Concatenate()([fc2, trend_class_input, trend_rank_input])
fc4 = Dropout(0.3)(fc3)
fc5 = Dense(32, activation='relu', kernel_initializer='glorot_uniform', kernel_constraint=maxnorm(3))(fc4)
#fc4 = Dropout(0.5)(fc3)
out = Dense(1)(fc5)

'''
nlp_input = Input(shape=(seq_length,), name='nlp_input')
meta_input = Input(shape=(10,), name='meta_input')
emb = Embedding(output_dim=embedding_size, input_dim=100, input_length=seq_length)(nlp_input)
nlp_out = Bidirectional(LSTM(128, dropout=0.3, recurrent_dropout=0.3, kernel_regularizer=regularizers.l2(0.01)))(emb)
x = concatenate([nlp_out, meta_input])
x = Dense(classifier_neurons, activation='relu')(x)
x = Dense(1, activation='sigmoid')(x)
model = Model(inputs=[nlp_input , meta_input], outputs=[x])
'''

# Set up path for saving model and history
filepath="/content/gdrive/My Drive/Colab Notebooks/Insight_Net_Align/models/model_nn_full-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
history = History()
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
callbacks_list = [checkpoint, history, reduce_lr]

# Create model and compile it
model_nn = Model([country_input, product_input, trend_class_input, trend_rank_input], out)

sgd = optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
nadam = optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)

adam = optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=2e-4, amsgrad=False, clipnorm=1) # clipvalue = 0.5 or 0.25 for exploding gradients

model_nn.compile(optimizer=adam, loss='mean_squared_error', metrics=['logcosh', 'mean_absolute_error', 'mean_squared_error','cosine_proximity'])
model_nn.summary()


#%%
# Run the NN model
if os.path.exists('/content/gdrive/My Drive/Colab Notebooks/Insight_Net_Align/models/model.hdf5'):
  model_nn = load_model('/content/gdrive/My Drive/Colab Notebooks/Insight_Net_Align/models/model_nn_reduced-05-0.6526.hdf5')
else:
  history_nn = model_nn.fit([train.location_id, train.product_id, train.export_trend_class, train.export_trend_pct_rank], train.export_val_std_all, batch_size=32, epochs=10, verbose=1, callbacks=callbacks_list)

  with open('/content/gdrive/My Drive/Colab Notebooks/Insight_Net_Align/models/history_nn_full', 'wb') as file_pi:
    pickle.dump(history_nn.history, file_pi)

  plt.plot(history_nn.history['loss'])
  plt.xlabel('Number of Epochs')
  plt.ylabel('Training Error')

#%%
# Load a model
model_nn = load_model('/content/gdrive/My Drive/Colab Notebooks/Insight_Net_Align/models/model_nn_trend_rank-03-0.6580.hdf5')

history_nn = pickle.load(open('/content/gdrive/My Drive/Colab Notebooks/Insight_Net_Align/models/history_nn_trend_rank', 'rb'))
history_nn

'''
# more technically correct if kept adding to file
history_nn = []
with (open('/content/gdrive/My Drive/Colab Notebooks/Insight_Net_Align/models/history_nn_reduced', 'rb')) as openfile:
    while True:
        try:
            history_nn.append(pickle.load(openfile))
        except EOFError:
            break
'''

#%%
# Plot model loss
plt.plot(history_nn['loss'])
plt.xlabel('Number of Epochs')
plt.ylabel('Training Error')

#%%
# Evaluate model_nn on 2008 - can set = and add callbacks in order to get history?
print(model_nn.metrics_names)
model_nn.evaluate([test.location_id, test.product_id, test.export_trend_pct_rank], test.export_val_std_all)

#%%
# Make select predictions using model_dot - probably need to un-normalize from minmax
predictions_dot = model_dot.predict([test.location_id.head(20), test.product_id.head(20)])

[print(predictions_dot[i], test.export_pct_std.iloc[i]) for i in range(0,20)]
