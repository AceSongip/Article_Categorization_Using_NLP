# -*- coding: utf-8 -*-
"""
Created on Thu May 19 12:26:26 2022

@author: aceso
"""
#%% Modules
import pandas as pd
import os 
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import datetime
import pickle
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from text_classification_lib import EDA, ModelConfig, Performance
from tensorflow.keras.utils import plot_model

# Constant
URL = "https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv"
TOKEN_PATH = os.path.join(os.getcwd(), "Saved", "token.json")
LOG_PATH = os.path.join(os.getcwd(), 'Log')
log_dir = os.path.join(LOG_PATH, datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
MODEL_PATH = os.path.join(os.getcwd(), "Saved", "model.h5")
ONEHOT_SAVEPATH = os.path.join(os.getcwd(), "Saved", "onehot.pkl")

#%% Exploratory Data Analysis

# Data Loading
df = pd.read_csv(URL)
category = df["category"] # There're 5 categories
text = df["text"]

# Data Cleaning
eda = EDA()
split_text = eda.split(text)

# Data Vectorization
token_text = eda.category_token(data=split_text, token_save_path=TOKEN_PATH, num_words=2000)
print(token_text[2])

# Data Sequence Padding
[np.shape(i) for i in token_text] # to check the maxlen of word in each text
# maxlen is 300
pad_text = eda.text_pad_sequence(token_text)

# Data Preprocessing (Target One Hot Encoding)
one = OneHotEncoder(sparse=False)
nb_categories = len(category.unique())
encoded_category = one.fit_transform(np.expand_dims(category, axis=-1))
pickle.dump(one, open(ONEHOT_SAVEPATH, "wb"))

# Split the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(pad_text, encoded_category, 
                                                    test_size=0.2, random_state=123)

# The model only accept 3D array as input
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

# Inverse the category
print(y_train[0]) #[0,0,0,1,0]
print(one.inverse_transform(np.expand_dims(y_train[0], axis=0))) # This one is sport

#%% Model Configuration
nb_categories = len(category.unique())

nn = ModelConfig()
model = nn.lstm_layer(nb_words=2000, nb_categories=nb_categories, nodes=64)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics="acc")

# Plot model architecture
plot_model(model)

tensorboard = TensorBoard(log_dir, histogram_freq=1)
estop = EarlyStopping(monitor="val_loss", patience=5)

model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test),
          callbacks=[tensorboard, estop])

#%% Model Evaluation and Analysis
predicted = np.empty([len(X_test), 5]) # 5 onehot columns
for i, test in enumerate(X_test):
    predicted[i,:] = model.predict(np.expand_dims(test, axis=0))
    
y_pred = np.argmax(predicted, axis=1) 
y_true = np.argmax(y_test, axis=1)

score = Performance()
result = score.evaluate(y_true, y_pred)

#%% Model Saving
model.save(MODEL_PATH)

