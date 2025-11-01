import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
from tensorflow.python.framework.ops import SymbolicTensor
from sklearn.preprocessing import MinMaxScaler
from keras.src import ops
from task2_1_a import load_data, common_sense_categories_acc
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def h_numerical_cs_mae(y_true, y_pred):
    """
    
    """
    # print(y_true)
    # print(y_pred)
    # tf.print(y_true)
    # tf.print(y_pred)
    # input(y_true)
    # tf.cast(y_true, dtype=tf.float32)
    # tf.cast(y_pred, dtype=tf.float32)
    true = y_true[..., 0]
    pred = y_pred[..., 0]

    diff = tf.abs(true - pred)
    wrap = (12/11) - diff
    cls = tf.minimum(diff, wrap)

    mae = tf.reduce_mean(tf.abs(cls))
    return mae

def m_numerical_cs_mae(y_true, y_pred):
    """
    
    """
    # print(y_true)
    # print(y_pred)
    # tf.print(y_true)
    # tf.print(y_pred)
    # input(y_true)
    # tf.cast(y_true, dtype=tf.float32)
    # tf.cast(y_pred, dtype=tf.float32)
    true = y_true[..., 0]
    pred = y_pred[..., 0]

    diff = tf.abs(true - pred)
    wrap = (60/59) - diff
    cls = tf.minimum(diff, wrap)

    mae = tf.reduce_mean(tf.abs(cls))
    return mae


if __name__ == "__main__":
    os.makedirs('saved_models', exist_ok=True)
    
    seed=42
    np.random.seed(seed)
    tf.random.set_seed(seed)
    keras.utils.set_random_seed(seed)

    X_train, y_train, X_val, y_val, X_test, y_test = load_data(seed=42)

    print(X_train.shape, X_val.shape, X_test.shape)

    # scale y_train to [0, 1]
    print(y_train)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(y_train)
    y_train = scaler.transform(y_train)
    y_val = scaler.transform(y_val)
    y_test = scaler.transform(y_test)
    print(y_train)

    img_rows, img_cols = X_train.shape[1], X_train.shape[2]
    input_shape = (img_rows, img_cols, 1)
    print(input_shape)

    # regression model with two outputs
    input_ = Input(shape=input_shape)
    conv1 = Conv2D(32, kernel_size=(3, 3), activation='relu')(input_)
    conv2 = Conv2D(64, (3, 3), activation='relu')(conv1)
    pooling = MaxPooling2D(pool_size=(2, 2))(conv2)
    dropout = Dropout(0.25)(pooling)
    flatten = Flatten()(dropout)
    hidden1 = Dense(128, activation='relu')(flatten)
    dropout =  Dropout(0.5)(hidden1)
    hour_output = Dense(1, activation='sigmoid')(dropout)
    minute_out = Dense(1, activation='sigmoid')(dropout)
    
    model = keras.Model(inputs=input_, outputs=[hour_output, minute_out])
    
    model.compile(loss=[h_numerical_cs_mae, m_numerical_cs_mae],
                optimizer=keras.optimizers.Adadelta(),
                metrics=['accuracy','accuracy'])

    model.summary()

    batch_size = 128
    epochs = 12

    model.fit(X_train, [y_train[:, 0], y_train[:, 1]],
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(X_val, [y_val[:, 0], y_val[:, 1]]))

    score = model.evaluate(X_test, [y_test[:, 0], y_test[:, 1]], verbose=0)
    print('Test loss:', score[0])
    print('Test hour loss:', score[1])
    print('Test minute loss:', score[2])
    print('Test hour accuracy:', score[3])
    print('Test minute accuracy:', score[4])

    model.save('saved_models/two_outputs_regression.keras')