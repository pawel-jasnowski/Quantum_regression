#!/usr/bin/env python
# coding: utf-8

# #Import modules

# In[2]:


import pandas as pd
import numpy as np
import sweetviz as sv

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# #Loading data / EDA / Data cleaning / Standarize

# In[3]:


raw_train_data = pd.read_csv('quantum/internship_train.csv')
raw_test_data = pd.read_csv('quantum/internship_hidden_test.csv')

# In[4]:


train_data = raw_train_data.copy()
test_data = raw_test_data.copy()

# In[5]:


print(len(train_data))
print(len(test_data))

# In[6]:


train_data.tail()

# In[7]:


train_data.isnull().sum()

# In[12]:


# my_report = sv.analyze(train_data)


# In[8]:


# my_report.show_html('my_report.html', open_browser=True)


# In[13]:


train_data.describe()

# In[11]:


train_data.info()

# In[11]:


# compare_report = sv.compare([train_data, 'Train'], [test_data, 'Test'], 'target')


# In[12]:


# compare_report.show_html('Compare_report.html')


# In[14]:


train_data.keys()

# In[15]:


train_target = train_data.pop('target')

# In[16]:


train_target.head()

# In[17]:


train_stats = train_data.describe()
train_stats = train_stats.transpose()
train_stats


# In[18]:


def norm(x):
    return (x - train_stats['mean']) / train_stats['std']


# In[19]:


normed_train_data = norm(train_data)
normed_test_data = norm(test_data)

# In[20]:


normed_test_data.head()

# In[21]:


normed_train_data.values
normed_test_data.values


# #Model Building

# In[22]:


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


# In[23]:


model = build_model()
model.summary()

# In[24]:


filepath = 'best_model_weights.hdf5'
checkpoint = ModelCheckpoint(filepath=filepath, monitor='mse', save_best_only=True, verbose=1)
es = EarlyStopping(monitor='mse', mode='min', verbose=1, patience=5)

# In[25]:


history = model.fit(normed_train_data, train_target, epochs=150, validation_split=0.2, verbose=1, batch_size=32,
                    callbacks=[checkpoint, es])

# In[26]:


metrics = pd.DataFrame(history.history)

# In[28]:


metrics.head(10)


# In[29]:


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

# In[30]:


test_predictions = model.predict(normed_test_data)

# In[31]:


test_predictions

# In[32]:


pred = pd.DataFrame(test_predictions, columns=['predictions'])
pred.describe()

# In[33]:


train_target.describe()

# In[ ]:




