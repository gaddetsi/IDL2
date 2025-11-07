import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input, BatchNormalization, Activation
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
from tensorflow.python.framework.ops import SymbolicTensor
from sklearn.preprocessing import MinMaxScaler
from keras.src import ops
from task2_1_a import load_data, to_categorical, common_sense_mse
import os

# uncomment if you want to use the cpu (needed for printing)
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

BATCH_SIZE = 128
EPOCHS = 1000 # high number since we use early stopping


@tf.keras.utils.register_keras_serializable()
def h_numerical_cs_mse(y_true, y_pred):
    """
    
    """
    true = y_true[..., 0]
    pred = y_pred[..., 0]

    diff = tf.abs(true - pred)
    wrap = 12 - diff
    cls = tf.minimum(diff, wrap)

    mse = tf.square(cls / 6)
    return mse

@tf.keras.utils.register_keras_serializable()
def m_numerical_cs_mse(y_true, y_pred):
    """
    
    """
    true = y_true[..., 0]
    pred = y_pred[..., 0]

    diff = tf.abs(true - pred)
    wrap = 60 - diff
    cls = tf.minimum(diff, wrap)

    mse = tf.square(cls / 30)
    return mse

@tf.keras.utils.register_keras_serializable()
def common_sense_mse_12(y_true, y_pred):
    """common sense mse from task 2.1.a with num_classes = 12"""
    return common_sense_mse(y_true, y_pred, num_classes=12)

def hour_labels_to_sin_cos(y):
    """
    Convert (hour, minute) → [cos, sin] of hour angle.
    This encoding handles the circular nature of time.
    """
    hours_float = y[:, 0] + y[:, 1] / 60.0
    angle = 2 * np.pi * hours_float / 12.0  # full rotation = 12 hours
    y_cos = np.cos(angle)
    y_sin = np.sin(angle)
    return np.stack([y_cos, y_sin], axis=1)  # shape: (N, 2)

def minute_labels_to_sin_cos(y):
    """
    Convert (hour, minute) → [cos, sin] of minute angle.
    This encoding handles the circular nature of time.
    """
    minutes_float = y[:, 1]
    angle = 2 * np.pi * minutes_float / 60.0  # full rotation = 60 minutes
    y_cos = np.cos(angle)
    y_sin = np.sin(angle)
    return np.stack([y_cos, y_sin], axis=1)  # shape: (N, 2)


def build_cnn_multi_class_reg(input_shape):
    """CNN with multi-headed regression (two paths for hours and minutes)."""
    inputs = Input(shape=input_shape)
    
    # First conv block
    x1 = Conv2D(32, (3, 3), padding='same')(inputs)
    x2 = Conv2D(32, (3, 3), padding='same')(inputs)
    x1 = BatchNormalization()(x1)
    x2 = BatchNormalization()(x2)
    x1 = Activation('relu')(x1)
    x2 = Activation('relu')(x2)
    x1 = Conv2D(32, (3, 3), padding='same')(x1)
    x2 = Conv2D(32, (3, 3), padding='same')(x2)
    x1 = BatchNormalization()(x1)
    x2 = BatchNormalization()(x2)
    x1 = Activation('relu')(x1)
    x2 = Activation('relu')(x2)
    x1 = MaxPooling2D((2, 2))(x1)
    x2 = MaxPooling2D((2, 2))(x2)
    x1 = Dropout(0.2)(x1)
    x2 = Dropout(0.2)(x2)
    
    # Second conv block
    x1 = Conv2D(64, (3, 3), padding='same')(x1)
    x2 = Conv2D(64, (3, 3), padding='same')(x2)
    x1 = BatchNormalization()(x1)
    x2 = BatchNormalization()(x2)
    x1 = Activation('relu')(x1)
    x2 = Activation('relu')(x2)
    x1 = Conv2D(64, (3, 3), padding='same')(x1)
    x2 = Conv2D(64, (3, 3), padding='same')(x2)
    x1 = BatchNormalization()(x1)
    x2 = BatchNormalization()(x2)
    x1 = Activation('relu')(x1)
    x2 = Activation('relu')(x2)
    x1 = MaxPooling2D((2, 2))(x1)
    x2 = MaxPooling2D((2, 2))(x2)
    x1 = Dropout(0.2)(x1)
    x2 = Dropout(0.2)(x2)

    # Third conv block
    x1 = Conv2D(128, (3, 3), padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = MaxPooling2D((2, 2))(x1)
    x1 = Dropout(0.3)(x1)

    x2 = Conv2D(128, (3, 3), padding='same')(x2)
    x2 = BatchNormalization()(x2)
    x2 = Activation('relu')(x2)
    x2 = MaxPooling2D((2, 2))(x2)
    x2 = Dropout(0.3)(x2)

    # Dense layers
    x1 = Flatten()(x1)
    x1 = Dense(128)(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = Dropout(0.4)(x1)

    x2 = Flatten()(x2)
    x2 = Dense(128)(x2)
    x2 = BatchNormalization()(x2)
    x2 = Activation('relu')(x2)
    x2 = Dropout(0.4)(x2)

    x1 = Flatten()(x1)
    x1 = Dense(256)(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = Dropout(0.4)(x1)

    x2 = Flatten()(x2)
    x2 = Dense(256)(x2)
    x2 = BatchNormalization()(x2)
    x2 = Activation('relu')(x2)
    x2 = Dropout(0.4)(x2)

    x1 = Dense(128)(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = Dropout(0.3)(x1)

    x2 = Dense(128)(x2)
    x2 = BatchNormalization()(x2)
    x2 = Activation('relu')(x2)
    x2 = Dropout(0.3)(x2)

    # two output layers for hours and minutes
    # hour has 12 output nodes for every hour class (0-11) with softmax activation
    hour_output = Dense(12, activation='softmax')(x1)
    minute_output = Dense(1, activation='linear')(x2)

    return keras.Model(inputs, [hour_output, minute_output], name="cnn_multi_class_regression_big")



def build_cnn_multi_sin_cos(input_shape):
    """CNN for periodic regression (predicting sin/cos of time angle)."""
    inputs = Input(shape=input_shape)
    
    # First conv block
    x1 = Conv2D(32, (3, 3), padding='same')(inputs)
    x2 = Conv2D(32, (3, 3), padding='same')(inputs)
    x1 = BatchNormalization()(x1)
    x2 = BatchNormalization()(x2)
    x1 = Activation('relu')(x1)
    x2 = Activation('relu')(x2)
    x1 = Conv2D(32, (3, 3), padding='same')(x1)
    x2 = Conv2D(32, (3, 3), padding='same')(x2)
    x1 = BatchNormalization()(x1)
    x2 = BatchNormalization()(x2)
    x1 = Activation('relu')(x1)
    x2 = Activation('relu')(x2)
    x1 = MaxPooling2D((2, 2))(x1)
    x2 = MaxPooling2D((2, 2))(x2)
    x1 = Dropout(0.2)(x1)
    x2 = Dropout(0.2)(x2)
    
    # Second conv block
    x1 = Conv2D(64, (3, 3), padding='same')(x1)
    x2 = Conv2D(64, (3, 3), padding='same')(x2)
    x1 = BatchNormalization()(x1)
    x2 = BatchNormalization()(x2)
    x1 = Activation('relu')(x1)
    x2 = Activation('relu')(x2)
    x1 = Conv2D(64, (3, 3), padding='same')(x1)
    x2 = Conv2D(64, (3, 3), padding='same')(x2)
    x1 = BatchNormalization()(x1)
    x2 = BatchNormalization()(x2)
    x1 = Activation('relu')(x1)
    x2 = Activation('relu')(x2)
    x1 = MaxPooling2D((2, 2))(x1)
    x2 = MaxPooling2D((2, 2))(x2)
    x1 = Dropout(0.2)(x1)
    x2 = Dropout(0.2)(x2)

    # Third conv block
    x1 = Conv2D(128, (3, 3), padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = MaxPooling2D((2, 2))(x1)
    x1 = Dropout(0.3)(x1)

    x2 = Conv2D(128, (3, 3), padding='same')(x2)
    x2 = BatchNormalization()(x2)
    x2 = Activation('relu')(x2)
    x2 = MaxPooling2D((2, 2))(x2)
    x2 = Dropout(0.3)(x2)

    # Dense layers
    x1 = Flatten()(x1)
    x1 = Dense(128)(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = Dropout(0.4)(x1)

    x2 = Flatten()(x2)
    x2 = Dense(128)(x2)
    x2 = BatchNormalization()(x2)
    x2 = Activation('relu')(x2)
    x2 = Dropout(0.4)(x2)

    x1 = Flatten()(x1)
    x1 = Dense(256)(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = Dropout(0.4)(x1)

    x2 = Flatten()(x2)
    x2 = Dense(256)(x2)
    x2 = BatchNormalization()(x2)
    x2 = Activation('relu')(x2)
    x2 = Dropout(0.4)(x2)

    x1 = Dense(128)(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = Dropout(0.3)(x1)

    x2 = Dense(128)(x2)
    x2 = BatchNormalization()(x2)
    x2 = Activation('relu')(x2)
    x2 = Dropout(0.3)(x2)

    # Output layer: 2 nodes for cos and sin, bounded by tanh
    hours_output = Dense(12, activation='softmax')(x1)
    minutes_output = Dense(2, activation='tanh')(x2)

    return keras.Model(inputs, [hours_output, minutes_output], name="multi_cnn_periodic")


if __name__ == "__main__":
    os.makedirs('saved_models', exist_ok=True)
    
    # multi class+regression
    seed=42
    np.random.seed(seed)
    tf.random.set_seed(seed)
    keras.utils.set_random_seed(seed)

    X_train, y_train, X_val, y_val, X_test, y_test, input_shape = load_data(seed=seed, easy=False)

    print(X_train.shape, X_val.shape, X_test.shape)

    y_train1 = to_categorical(y_train, 12)
    y_val1 = to_categorical(y_val, 12)
    y_test1 = to_categorical(y_test, 12)

    img_rows, img_cols = X_train.shape[1], X_train.shape[2]
    input_shape = (img_rows, img_cols, 1)
    print(input_shape)

    # regression model with two outputs
    model = build_cnn_multi_class_reg(input_shape)

    model.compile(loss=[common_sense_mse_12, "mse"],
                loss_weights=[1.0, 0.5],
                optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                metrics=['accuracy','accuracy'])

    model.summary()

    # Callbacks
    callbacks = [
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', patience=5, factor=0.5, verbose=1, min_lr=1e-7
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=15, verbose=1, restore_best_weights=True
        )
    ]

    model.fit(X_train, [y_train1, y_train[:, 1]],
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            verbose=1,
            callbacks=callbacks,
            validation_data=(X_val, [y_val1, y_val[:, 1]]))

    score = model.evaluate(X_test, [y_test1, y_test[:, 1]], verbose=0)
    print('Test loss:', score[0])
    print('Test hour loss:', score[1])
    print('Test minute loss:', score[2])
    print('Test hour accuracy:', score[3])
    print('Test minute accuracy:', score[4])

    model.save('saved_models/multi_class_regression.keras')

    # periodic regression
    seed=42
    np.random.seed(seed)
    tf.random.set_seed(seed)
    keras.utils.set_random_seed(seed)

    X_train, y_train, X_val, y_val, X_test, y_test, input_shape = load_data(seed=42, easy=False)

    print(X_train.shape, X_val.shape, X_test.shape)

    y_train_m = minute_labels_to_sin_cos(y_train)
    y_val_m = minute_labels_to_sin_cos(y_val)
    y_test_m = minute_labels_to_sin_cos(y_test)

    y_train_h = to_categorical(y_train, 12)
    y_val_h = to_categorical(y_val, 12)
    y_test_h = to_categorical(y_test, 12)

    img_rows, img_cols = X_train.shape[1], X_train.shape[2]
    input_shape = (img_rows, img_cols, 1)
    print(input_shape)

    # regression model with two outputs
    model = build_cnn_multi_sin_cos(input_shape)

    model.compile(loss=[common_sense_mse_12, "mse"],
                loss_weights=[1.0, 0.5],
                optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                metrics=['accuracy','accuracy'])

    model.summary()

    # Callbacks
    callbacks = [
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', patience=5, factor=0.5, verbose=1, min_lr=1e-7
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=15, verbose=1, restore_best_weights=True
        )
    ]

    model.fit(X_train, [y_train_h, y_train_m],
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            verbose=1,
            callbacks=callbacks,
            validation_data=(X_val, [y_val_h, y_val_m]))

    score = model.evaluate(X_test, [y_test_h, y_test_m], verbose=0)
    print('Test loss:', score[0])
    print('Test hour loss:', score[1])
    print('Test minute loss:', score[2])
    print('Test hour accuracy:', score[3])
    print('Test minute accuracy:', score[4])

    model.save('saved_models/multi_regression_sin_cos.keras')