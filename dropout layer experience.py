import tensorflow as tf
import numpy as np
# Dropout class
# Applies Dropout to the input
# The Dropout layer randomly sets input units to 0 with a frequency of rate at each step
# during training time, which helps prevent overfitting. Inputs not set to 0 are selected
# up by 1/(1-rate) such that the sum over all inputs is unchanged.

# Note that the Dropout layer only applies when training is set to True such that no values
# are dropped during inference. When using model.fit, training will be appropriately set to
# True automatically, and in other contexts, you can set the kwarg explicitly to True
# when calling the layer.

# (This is in contrast to setting trainable=False for a Dropout layer. trainable does not)
# affect the layer's behavior, as Dropout does not have any variables/weights that can be frozen
# during training.

tf.radom.set_seed(0)
layer = tf.keras.layers.Dropout(.2,input_shape=(2,))
data = np.arange(10).reshape(5,2).astype(np.float32)
outputs = layer(data, training=True)
print(outputs)
# Then we can get a tensor variable
# But it shows on my console that
# "This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN)
# to use the following CPU instruct to enable them in
# performance-critical operatioins: AVX AVX2"
# So maybe I need to check the suitable version of tensorflows and then set my environnment properly

# What is this for is a regularization technique used in deep
# learning to prevent overfitting.
# Randomly "drops out" or sets during training, this means that the neurons are not in the layer
# during training. This means that the neurons are not used to make predictions, and their
# activations are not updated. The proportion of neurons dropped out is specified as a hyperparameter and
# is usually set to a value between 0.2 to 0.5

# We often adjust this parameter in the training layer but deactivated in test process
# tensorflow reference: https://www.tensorflow.org/

# And an important part of Tensorflow is that it is supposed to be fast. With a suitable
# installation, it works with CPUs, GPUs, or TPUs. Part of going fast means that it uses
# different code depending on your hardware. Some CPUs support operations that other CPUs
# do not, such as vectorized addition (adding multiple variables at once). Tensorflow is simply
# telling you that the version you have installed can use the AVX adn AVX2 operations and is
# set to do so by default in certain situations(say inside a forward or back-prop matrix multiply),
# which can speed things up. This is not an error, it is just telling you that it can and will take
# advantage of your CPU to get that extra speed out.

