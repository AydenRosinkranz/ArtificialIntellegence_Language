import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
# os.environ['CUDA_VISIBLE_DEVICES']=""
import tensorflow as tf
BATCH_SIZE = 16
imdb_train = tf.keras.utils.text_dataset_from_directory(
    "D:/anaconda3/aclImdb/train",
    batch_size = BATCH_SIZE,
)
imdb_test = tf.keras.utils.text_dataset_from_directory(
    "D:/anaconda3/aclImdb/test",
    batch_size=BATCH_SIZE,
)

# Inspect first review
# Format is (review text tensor, label tensor)
print(imdb_train.unbatch().take(1).get_single_element())