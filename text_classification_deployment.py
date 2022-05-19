# -*- coding: utf-8 -*-
"""
Created on Thu May 19 15:40:07 2022

How to use this? 
Just copy any article text about politics, sport,entertainment, busisness or tech 
and feed into input in the console

@author: aceso
"""

import os
import json
import numpy as np
from text_classification_lib import EDA, ModelConfig
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.models import load_model
import pickle
import warnings

warnings.filterwarnings("ignore") # supress warning

# Constant
MODEL_PATH = os.path.join(os.getcwd(), "Saved", "model.h5")
TOKEN_PATH = os.path.join(os.getcwd(), "Saved", "token.json")
ONEHOT_SAVEPATH = os.path.join(os.getcwd(), "Saved", "onehot.pkl")

#%% Loading

# Model loading
model = load_model(MODEL_PATH)

# Token loading
with open(TOKEN_PATH, "r") as json_file:
    token = json.load(json_file)
# Encoder loading
with open(ONEHOT_SAVEPATH, "rb") as r:
    one = pickle.load(r)
    
#%% Example

new_text = [input("Please input the text: ")]

#%% Clean the data
eda = EDA()
clean_text = eda.split(new_text)

#%% Data preprocessing
loaded_token = tokenizer_from_json(token)
clean_text = loaded_token.texts_to_sequences(clean_text)
clean_text = eda.text_pad_sequence(clean_text)

#%% Model prediction
pred = model.predict(np.expand_dims(clean_text, axis=-1))
pred = one.inverse_transform(pred)
print(f"The prediction for the input text is {pred}")


