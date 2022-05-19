# -*- coding: utf-8 -*-
"""
Created on Thu May 19 10:38:24 2022

@author: aceso
"""
#%% Modules
import pandas as pd
import numpy as np
import re
import os
import json
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Embedding, Bidirectional
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Constant
URL = "https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv"
TOKEN_PATH = os.path.join(os.getcwd(),"Saved" ,"token.json")
#%% Classes

# Exploratory Data Analysis class
class EDA():
    
    # splitting the text and remove numerical, no need lowercase
    def split(self, data):
        for i, text in enumerate(data):
            data[i] = re.sub("[^a-zA-Z]", " ", text).split()
            
        return data
    
    # tokenization
    # build token
    def category_token(self, data, token_save_path, num_words, oov_token="<oov>"):
        tokenizer = Tokenizer(num_words, oov_token=(oov_token))
        tokenizer.fit_on_texts(data)
        # save token
        token_json = tokenizer.to_json()
        with open(TOKEN_PATH, "w") as json_file:
            json.dump(token_json, json_file)
        word_index = tokenizer.word_index
        print(word_index)
        
        return tokenizer.texts_to_sequences(data)
    
    # pad sequence
    def text_pad_sequence(self, data, maxlen=300):
        return pad_sequences(data, maxlen, padding="post", truncating="post")
    
# Model Building class
class ModelConfig():
    
    def lstm_layer(self, nb_words, nb_categories, nodes, embedding_output=64):
        model = Sequential()
        model.add(Embedding(nb_words, embedding_output))
        model.add(Bidirectional(LSTM(nodes, return_sequences=(True))))
        model.add(Dropout(0.2))
        model.add(Bidirectional(LSTM(nodes, return_sequences=(True))))
        model.add(Dropout(0.2))
        model.add(Bidirectional(LSTM(nodes)))
        model.add(Dense(nb_categories, activation="softmax"))
        model.summary()
        
        return model

# Model Performance Evaluation class
class Performance():
    
    def evaluate(self, y_true, y_pred):
        print(classification_report(y_true, y_pred)) # classification report
        print(confusion_matrix(y_true, y_pred)) # confusion matrix
        print(accuracy_score(y_true, y_pred))
        

    
#%% Testing
if __name__ == "__main__":
    # Data Loading
    df = pd.read_csv(URL)
    category = df["category"] # There're 5 categories
    text = df["text"]
    
    # Data Cleaning
    eda = EDA()
    split_text = eda.split(text)
    
    # Tokenization
    token_text = eda.category_token(data=split_text, token_save_path=TOKEN_PATH, num_words=2000)
    print(token_text[2])
    
    # Padding
    [np.shape(i) for i in token_text] # to check the maxlen of word in each text
    # maxlen is 300
    pad_text = eda.text_pad_sequence(token_text, maxlen=300) # after 300 words will be chop
