import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

max_features = 20000 # Only consider the top 20k words
maxlen = 200 # Only consider the first 200 words of each movie review

inputs = keras.Input(shape=(None,),dtype="int32")
x = layers.Embedding(max_features,128)(inputs)
# Add 2 bidirectional LSTMs
x = layers.Bidirectional(layers.LSTM(64,return_sequences=True))(x)
x = layers.Bidirectional(layers.LSTM(64))(x)

outputs = layers.Dense(1,activation='sigmoid')(x)
model = keras.Model(inputs,outputs)
model.summary()

