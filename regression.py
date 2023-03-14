#!/usr/bin/env python
# coding: utf-8

# #Import modules

# In[1]:


import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# #Loading data / EDA / Data cleaning / Standarize

# In[2]:


raw_train_data = pd.read_csv('quantum/internship_train.csv')
raw_test_data = pd.read_csv('quantum/internship_hidden_test.csv')

# In[3]:


train_data = raw_train_data.copy()
test_data = raw_test_data.copy()

# In[4]:


print(len(train_data))
print(len(test_data))

# In[5]:


train_data.tail()

# In[6]:


train_data.isnull().sum()

# In[7]:


train_data.describe()

# In[8]:


train_data.info()

# In[9]:


train_data.keys()

# In[10]:


train_target = train_data.pop('target')

# In[11]:


train_target.head()

# In[12]:


train_stats = train_data.describe()
train_stats = train_stats.transpose()
train_stats


# In[13]:


def norm(x):  # normalization
    return (x - train_stats['mean']) / train_stats['std']


# In[14]:


normed_train_data = norm(train_data)
normed_test_data = norm(test_data)

# In[15]:


normed_test_data.head()

# In[16]:


normed_train_data.values
normed_test_data.values


# #Model Building

# In[17]:


def build_model():
    model = Sequential()
    model.add(Dense(512, kernel_regularizer='l2', activation='relu', input_shape=[len(train_data.keys())]))
    model.add(Dense(128, activation='relu'))
    # model.add(Dense(32, activation='relu'))
    model.add(Dense(1))

    model.compile(optimizer='adam',
                  loss='mse',
                  metrics=['mae', 'mse'])
    return model


# In[18]:


model = build_model()
model.summary()

# In[19]:


filepath = 'best_model_weights.hdf5'
checkpoint = ModelCheckpoint(filepath=filepath, monitor='mse', save_best_only=True, verbose=1)
es = EarlyStopping(monitor='mse', mode='min', verbose=1, patience=5)

# In[ ]:


history = model.fit(normed_train_data, train_target, epochs=150, validation_split=0.2, verbose=1, batch_size=32,
                    callbacks=[checkpoint, es])

# In[ ]:


metrics = pd.DataFrame(history.history)

# In[ ]:


metrics.head(10)


# In[ ]:


def plot_hist(metrics):
    metrics['epoch'] = history.epoch
    metrics['rmse'] = np.sqrt(metrics['mse'])
    metrics['val_rmse'] = np.sqrt(metrics['val_mse'])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=metrics['epoch'], y=metrics['mae'], name='mae', mode='markers+lines'))
    fig.add_trace(go.Scatter(x=metrics['epoch'], y=metrics['val_mae'], name='val_mae', mode='markers+lines'))
    fig.update_layout(width=800, height=500, title='MAE vs VAL_MAE', xaxis_title='epochs', yaxis_title='MAE')
    fig.show()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=metrics['epoch'], y=metrics['rmse'], name='rmse', mode='markers+lines'))
    fig.add_trace(go.Scatter(x=metrics['epoch'], y=metrics['val_rmse'], name='val_rmse', mode='markers+lines'))
    fig.update_layout(width=800, height=500, title='RMSE vs VAL_RMSE', xaxis_title='epochs', yaxis_title='RMSE')
    fig.show()


plot_hist(metrics)

# #Prediction

# In[ ]:


test_predictions = model.predict(normed_test_data)

# In[ ]:


test_predictions

# In[ ]:


pred = pd.DataFrame(test_predictions, columns=['predictions'])
pred.describe()

# In[ ]:


train_target.describe()

# In[ ]:


type(test_predictions)

# In[ ]:


test_target = pd.DataFrame(test_predictions)
test_target

# In[ ]:


test_target.to_csv('target_for_traning', header=None)

