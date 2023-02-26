import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as layer
import numpy as np

# Input Embeddings
class TokenAndPositionEmbedding(layer.Layer):
    def __init__(self,maxlen, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = layer.Embedding(input_dim=vocab_size,output_dim=embed_dim)
        self.pos_emb = layer.Embedding(input_dim=maxlen, output_dim=embed_dim)
    def call(self,x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions
# Encoder
# Multi-head self-attention
num_heads = 2
# Position-wise feed-forward network
ff_dim = 32
# Residual connections and layer normalization
# Decoder
# Masked
# Multi-head attention over encoder output
class TransformerBlock(layer.Layer):
    def __init__(self,embed_dim,num_heads,ff_dim,rate=0.1):
        super().__init__()
        self.att = layer.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layer.Dense(ff_dim, activation="relu"), layer.Dense(embed_dim), ]
        )
        self.layernorm1 = layer.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layer.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layer.Dropout(rate)
        self.dropout2 = layer.Dropout(rate)

    def call(self,inputs,training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
# Output projection

N = 20000
d = 200
(x_train, y_train), (x_val, y_val) = keras.datasets.imdb.load_data(num_words=N)
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=d)
x_val = keras.preprocessing.sequence.pad_sequences(x_val, maxlen=d)

inputs = layer.Input(shape=(d,))
embed_dim = 32
embedding_layer = TokenAndPositionEmbedding(d,N, embed_dim)
x = embedding_layer(inputs)
# print(x)
x = layer.GlobalAveragePooling1D()(x)
# print(x)
x = layer.Dropout(0.1)(x)
x = layer.Dense(20, activation="relu")(x)
x = layer.Dropout(0.1)(x)
outputs = layer.Dense(2, activation="softmax")(x)

model = keras.Model(inputs=inputs, outputs=outputs)

# Compilation
model.compile(
    optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"]
)
history = model.fit(x_train,y_train,batch_size=32,epochs=2,validation_data=(x_val,y_val))