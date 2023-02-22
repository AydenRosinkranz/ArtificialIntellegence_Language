import tensorflow as tf
input_shape=(4,7,10,128)
x = tf.random.normal(input_shape)
y = tf.keras.layers.Conv1D(
    32,3,activation='relu',input_shape=input_shape[1:]
)(x)
print(y.shape)