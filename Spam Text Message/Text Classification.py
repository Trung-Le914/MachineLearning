#!/usr/bin/env python
# coding: utf-8

# DATA: https://www.kaggle.com/datasets/team-ai/spam-text-message-classification?fbclid=IwZXh0bgNhZW0CMTAAAR0Lvy5i3S3kE6uod7LH6QGOYVamvHeXiwGi419H0Vn-z6ZD5JPcXVCBbvE_aem_VYil062imkKyQOrnK69SLg

# **IMPORT LIBRARIES**

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import string
import nltk
nltk.download('stopwords')
nltk.download('punkt')

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


# **LOADING THE DATASET**

# In[2]:


data_path = '.../Data.csv'
df = pd.read_csv(data_path)

messages = df['Message'].values.tolist()
labels = df['Category'].values.tolist()


# In[3]:


df.head()


# **PREPROCESSING DATA**

# In[4]:


def preprocess_text(text):
    
    # Lowering case
    text = text.lower()
    
    # Punctuation removal
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    
    # Tokenization
    tokens = nltk.word_tokenize(text)
    
    # Remove stopwords
    stop_words = nltk.corpus.stopwords.words('english')
    tokens = [token for token in tokens if token not in stop_words]
    
    # Stemming
    stemmer = nltk.PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]
    
    return tokens


# In[5]:


messages = [preprocess_text(message) for message in messages]


# **CREATE DICTIONARY**

# In[6]:


def create_dictionary(messages):
    dictionary = []
    
    for tokens in messages:
        for token in tokens:
            if token not in dictionary:
                dictionary.append(token)
                
    return dictionary


# In[7]:


dictionary = create_dictionary(messages)


# **CREATE FEATURES**

# In[8]:


def create_features(tokens, dictionary ):
    features = np.zeros(len(dictionary))

    for token in tokens :
        if token in dictionary :
            features[dictionary.index(token)] += 1
        
    return features

X = np.array([create_features(tokens, dictionary) for tokens in messages])


# **PREPROCESSING LABELS**

# In[9]:


label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)

print(f'Classes: {label_encoder.classes_}')
print(f'Encoded labels: {y}')


# **TRAIN/TEST SPLIT**

# In[10]:


TEST_SIZE = 0.125
VAL_SIZE = 0.2
SEED = 42


# In[11]:


X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=VAL_SIZE,
    shuffle=True, random_state=SEED
)

X_train.shape, X_val.shape, y_train.shape, y_val.shape


# In[12]:


X_train, X_test, y_train, y_test = train_test_split(
    X_train, y_train, test_size=TEST_SIZE,
    shuffle=True, random_state=SEED
)

X_train.shape, X_test.shape, y_train.shape, y_test.shape


# **TRAINING THE MODEL**

# In[13]:


model = GaussianNB()
print('Start training ...')
model = model.fit(X_train, y_train)
print('Training completed!')


# **EVALUATION**

# In[14]:


y_val_pred = model.predict(X_val)
y_test_pred = model.predict(X_test)

val_accuracy = accuracy_score(y_val, y_val_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f'Val accuracy: {val_accuracy}')
print(f'Test accuracy: {test_accuracy}')


# **PREDICTION**

# In[15]:


def predict (text, model, dictionary):
    processed_text = preprocess_text(text)
    features = create_features(text, dictionary)
    features = np.array(features).reshape(1, -1)
    prediction = model.predict(features)
    
    # We use inverse_transform to convert the variable to be 'ham' or 'Spam'
    prediction_cls = label_encoder.inverse_transform(prediction)[0]
    
    return prediction_cls

test_input = 'I am actually thinking a way of doing something useful '
prediction_cls = predict(test_input, model, dictionary)
print(f'Prediction: {prediction_cls}')


# In[ ]:




