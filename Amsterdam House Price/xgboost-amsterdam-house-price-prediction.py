#!/usr/bin/env python
# coding: utf-8

# DATA: https://www.kaggle.com/datasets/thomasnibb/amsterdam-house-price-prediction

# **IMPORT LIBRARIES**

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# In[2]:


# Read dataset
data_path = '/kaggle/input/amsterdam-house-price-prediction/HousingPrices-Amsterdam-August-2021.csv'
df = pd.read_csv(data_path)

# We drop redundant columns
df = df.drop(columns = ['Unnamed: 0', 'Address', 'Zip'], axis = 1)
df.head()


# **PREPROCESS THE DATA**

# In[3]:


# Check NaN values
print(df.isnull().sum())
df = df.dropna()


# In[4]:


# Standardize the data
scaler = MinMaxScaler()
cols_to_normalize = ['Price', 'Area', 'Room', 'Lon', 'Lat']

for col in cols_to_normalize:
    df[col] = scaler.fit_transform(df[[col]])


# **TRAIN/TEST DATA SPLIT**

# In[5]:


# Input/Target split
X = df.drop(columns = ['Price'])
y = df['Price']

# Train/Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.3, random_state = 42
)

X_train.shape, X_test.shape, y_train.shape, y_test.shape


# **XGBOOST MODEL**

# In[6]:


# We implement XGBoost model
xg_reg = xgb.XGBRegressor(
    seed = 42,
    learning_rate = 0.01,
    n_estimators = 102,
    max_depth = 3
)

xg_reg.fit(X_train, y_train)


# In[7]:


# Predictions based on the model
predictions = xg_reg.predict(X_test)


# In[8]:


# Evaluate the model
def evaluation(data):
    predictions = xg_reg.predict(data)
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    print('Evaluation results on test set:')
    print(f'Mean Absolute Error: {mae}')
    print(f'Mean Squared Error: {mse}')

evaluation(X_test)


# In[ ]:




