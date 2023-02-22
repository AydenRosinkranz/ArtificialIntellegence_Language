import numpy as np
import tensorflow as tf
from d2l import tensorflow as d2l

def show_heatmaps(matrices, xlabel, ylabel, titles=None, figsize=(2.5,2.5),
                  cmap='Reds'):
    d2l.use_svg_display()
    num_rows, num_cols, _, _ = matrices.shape
    fig, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize,
                                 sharex=True,sharey=True,squeeze=False)
    for i,(row_axes, row_matrices) in enumerate(zip(axes, matrices)):
        for j, (ax, matrix) in enumerate(zip(row_axes, row_matrices)):
            pcm = ax.imshow(matrix.numpy(),cmap=cmap)
            if i==num_rows-1:
                ax.set_ylabel(ylabel)
            if titles:
                ax.set_title(titles[j])
    fig.colorbar(pcm,ax=axes,shrink=0.6)

attention_weights = tf.reshape(tf.eye(10),(1,1,10,10))
show_heatmaps(attention_weights,xlabel='Keys',ylabel='Queries')

# Define some kernels
def gaussian(x):
    return tf.exp(-x**2/2)
def boxcar(x):
    return tf.abs(x)<1.0
def constant(x):
    return 1.0+0*x
def epanechikov(x):
    return tf.maximum(1-tf.abs(x),0)

d2l.use_svg_display()
kernels = (gaussian, boxcar, constant, epanechikov)
names = ('Gaussian','Boxcar','Constant','Epanechikov')
fig, axes = d2l.plt.subplots(1, 4, figsize=(12,3),
                                 sharex=True,sharey=True,squeeze=False)
x = tf.range(-2.5,2.5,0.1)
for kernel, name, ax in zip(kernels, names, axes):
    ax.plot(x.numpy(),kernel(x).numpy())
    ax.set_xlabel(name)
def f(x):
    return 2*tf.sin(x)+x
n = 40
x_train = tf.sort(tf.random.uniform((n,1))*5,0)
y_train = f(x_train)+tf.random.normal((n,1))
x_val = tf.range(0,5,0.1)
y_val = f(x_val)

def nadaraya_watson(x_train, y_train, x_val, kernel):
    dists = tf.reshape(x_train, (-1, 1)) - tf.reshape(x_val, (1, -1))
    # Each column/row corresponds to each query/key
    k = tf.cast(kernel(dists), tf.float32)
    # Normalization over keys for each query
    attention_w = k / tf.reduce_sum(k, 0)
    y_hat = tf.transpose(tf.transpose(y_train)@attention_w)
    return y_hat, attention_w

def plot(x_train, y_train, x_val, y_val, kernels, names, attention=False):
    fig, axes = d2l.plt.subplots(1, 4, sharey=True, figsize=(12, 3))
    for kernel, name, ax in zip(kernels, names, axes):
        y_hat, attention_w = nadaraya_watson(x_train, y_train, x_val, kernel)
        if attention:
            pcm = ax.imshow(attention_w.numpy(), cmap='Reds')
        else:
            ax.plot(x_val, y_hat)
            ax.plot(x_val, y_val, 'm--')
            ax.plot(x_train, y_train, 'o', alpha=0.5);
        ax.set_xlabel(name)
        if not attention:
            ax.legend(['y_hat', 'y'])
    if attention:
        fig.colorbar(pcm, ax=axes, shrink=0.7)

plot(x_train, y_train, x_val, y_val, kernels, names)

plot(x_train, y_train, x_val, y_val, kernels, names, attention=True)

sigmas = (0.1, 0.2, 0.5, 1)
names = ['Sigma ' + str(sigma) for sigma in sigmas]

def gaussian_with_width(sigma):
    return (lambda x: tf.exp(-x**2 / (2*sigma**2)))

kernels = [gaussian_with_width(sigma) for sigma in sigmas]
plot(x_train, y_train, x_val, y_val, kernels, names)

plot(x_train, y_train, x_val, y_val, kernels, names, attention=True)

# Now, let's look at attention scoring functions

import tensorflow as tf
from d2l import tensorflow as d2l

def masked_softmax(X, valid_lens):  #@save
    """Perform softmax operation by masking elements on the last axis."""
    # X: 3D tensor, valid_lens: 1D or 2D tensor
    def _sequence_mask(X, valid_len, value=0):
        maxlen = X.shape[1]
        mask = tf.range(start=0, limit=maxlen, dtype=tf.float32)[
            None, :] < tf.cast(valid_len[:, None], dtype=tf.float32)

        if len(X.shape) == 3:
            return tf.where(tf.expand_dims(mask, axis=-1), X, value)
        else:
            return tf.where(mask, X, value)

    if valid_lens is None:
        return tf.nn.softmax(X, axis=-1)
    else:
        shape = X.shape
        if len(valid_lens.shape) == 1:
            valid_lens = tf.repeat(valid_lens, repeats=shape[1])

        else:
            valid_lens = tf.reshape(valid_lens, shape=-1)
        # On the last axis, replace masked elements with a very large negative
        # value, whose exponentiation outputs 0
        X = _sequence_mask(tf.reshape(X, shape=(-1, shape[-1])), valid_lens,
                           value=-1e6)
        return tf.nn.softmax(tf.reshape(X, shape=shape), axis=-1)
