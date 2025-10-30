import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
from tensorflow.python.framework.ops import SymbolicTensor
from keras.src import ops

def numerical_cs_mae(y_true, y_pred):
    """
    
    """
    true = y_true[..., 0]
    pred = y_pred[..., 0]

    diff = tf.abs(true - pred)
    wrap = (60/59) - diff
    cls = tf.minimum(diff, wrap)

    mae = tf.reduce_mean(tf.abs(cls))
    return mae

tf.cast([[1,0.5]], dtype=tf.float32)
a = numerical_cs_mae(tf.cast([[1]], dtype=tf.float32), tf.cast([[1]], dtype=tf.float32))
print(a.numpy())
a = numerical_cs_mae(tf.cast([[0]], dtype=tf.float32), tf.cast([[1]], dtype=tf.float32))
print(a.numpy())
a = numerical_cs_mae(tf.cast([[0.5]], dtype=tf.float32), tf.cast([[1]], dtype=tf.float32))
print(a.numpy())
a = numerical_cs_mae(tf.cast([[0.2]], dtype=tf.float32), tf.cast([[1]], dtype=tf.float32))
print(a.numpy())
a = numerical_cs_mae(tf.cast([[0.8]], dtype=tf.float32), tf.cast([[1]], dtype=tf.float32))
print(a.numpy())
a = numerical_cs_mae(tf.cast([[0.2]], dtype=tf.float32), tf.cast([[0.6]], dtype=tf.float32))
print(a.numpy())
a = numerical_cs_mae(tf.cast([[0.3]], dtype=tf.float32), tf.cast([[0.6]], dtype=tf.float32))
print(a.numpy())

