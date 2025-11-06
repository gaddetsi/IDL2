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

def split_to_diff_min(pred_time, true_time):
    """Calculate absolute difference in minutes between predicted and true times."""
    # if hours are one-hot encoded, convert to numerical
    hours: np.ndarray = pred_time[0]
    minutes: np.ndarray = pred_time[1]
    true_hours: np.ndarray = true_time[0]
    true_minutes: np.ndarray = true_time[1]
    if hours.shape[1] == 12:
        hours = np.argmax(hours, axis=1).reshape(-1, 1)

    # convert time to total minutes
    pred_total_min = (hours * 60) + minutes
    true_total_min = (true_hours * 60) + true_minutes

    # common sense loss in minutes
    diff_min = np.abs(pred_total_min - true_total_min)
    csl = np.minimum(diff_min, 720 - diff_min)

    return csl

def print_metrics(pred_time, true_time, model_name):
    """Print comprehensive evaluation metrics."""
    diff_min = split_to_diff_min(pred_time, true_time)

    mean_err = np.mean(diff_min)
    median_err = np.median(diff_min)
    std_err = np.std(diff_min)
    max_err = np.max(diff_min)
    
    within_5 = np.mean(diff_min <= 5) * 100
    within_10 = np.mean(diff_min <= 10) * 100
    within_15 = np.mean(diff_min <= 15) * 100
    within_30 = np.mean(diff_min <= 30) * 100
    
    print(f"\n{'=' * 80}")
    print(f"{model_name} - TEST SET RESULTS")
    print(f"{'=' * 80}")
    print(f"Mean Absolute Error:    {mean_err:.2f} minutes")
    print(f"Median Absolute Error:  {median_err:.2f} minutes")
    print(f"Std Deviation:          {std_err:.2f} minutes")
    print(f"Max Error:              {max_err:.2f} minutes")
    print(f"\nAccuracy within thresholds:")
    print(f"  Within 5 minutes:     {within_5:.1f}%")
    print(f"  Within 10 minutes:    {within_10:.1f}%")
    print(f"  Within 15 minutes:    {within_15:.1f}%")
    print(f"  Within 30 minutes:    {within_30:.1f}%")
    
    return {
        'mean': mean_err,
        'median': median_err,
        'std': std_err,
        'max': max_err,
        'within_5': within_5,
        'within_10': within_10,
        'within_15': within_15,
        'within_30': within_30,
        'predictions': pred_time,
        'errors': diff_min
    }

@tf.keras.utils.register_keras_serializable()
def h_numerical_cs_mae(y_true, y_pred):
    """
    
    """
    true = y_true[..., 0]
    pred = y_pred[..., 0]

    diff = tf.abs(true - pred)
    wrap = (12/11) - diff
    cls = tf.minimum(diff, wrap)

    mae = tf.reduce_mean(tf.abs(cls))
    return mae

@tf.keras.utils.register_keras_serializable()
def m_numerical_cs_mae(y_true, y_pred):
    """
    
    """
    true = y_true[..., 0]
    pred = y_pred[..., 0]

    diff = tf.abs(true - pred)
    wrap = (60/59) - diff
    cls = tf.minimum(diff, wrap)

    mae = tf.reduce_mean(tf.abs(cls))
    return mae

@tf.keras.utils.register_keras_serializable()
def common_sense_mse_cr(y_true, y_pred):
    """common sense mse from task 2.1.a with num_classes = 12"""
    return common_sense_mse(y_true, y_pred, num_classes=12)

def build_cnn_multi(input_shape):
    """CNN with multi-headed regression (two outputs for hours and minutes)."""
    inputs = Input(shape=input_shape)
    
    # First conv block
    x = Conv2D(32, (3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.2)(x)
    
    # Second conv block
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.2)(x)

    # Third conv block
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.3)(x)

    # Dense layers
    x = Flatten()(x)
    x = Dense(256)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.4)(x)

    x = Dense(128)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.3)(x)

    # two output layers for hours and minutes
    hour_output = Dense(1, activation='linear')(x)
    minute_output = Dense(1, activation='linear')(x)

    return keras.Model(inputs, [hour_output, minute_output], name="cnn_multi_regression")

def build_cnn_multi_class_reg(input_shape):
    """CNN with multi-headed regression (two outputs for hours and minutes)."""
    inputs = Input(shape=input_shape)
    
    # First conv block
    x = Conv2D(32, (3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.2)(x)
    
    # Second conv block
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.2)(x)

    # Third conv block
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.3)(x)

    # Dense layers
    x = Flatten()(x)
    x = Dense(256)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.4)(x)

    x = Dense(128)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.3)(x)

    # two output layers for hours and minutes
    # hour has 12 output nodes for every hour class (0-11) with softmax activation
    hour_output = Dense(12, activation='softmax')(x)
    minute_output = Dense(1, activation='linear')(x)

    return keras.Model(inputs, [hour_output, minute_output], name="cnn_multi_class_regression")



def build_cnn_sin_cos(input_shape):
    """CNN for periodic regression (predicting sin/cos of time angle)."""
    inputs = Input(shape=input_shape)
    
    # First conv block
    x = Conv2D(32, (3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.2)(x)

    # Second conv block
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.2)(x)

    # Third conv block
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.3)(x)

    # Dense layers
    x = Flatten()(x)
    x = Dense(256)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.4)(x)

    x = Dense(128)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.3)(x)

    # Output layer: 2 nodes for cos and sin, bounded by tanh
    outputs = Dense(2, activation='tanh')(x)

    return keras.Model(inputs, outputs, name="cnn_periodic")


if __name__ == "__main__":
    os.makedirs('saved_models', exist_ok=True)
    
    # seed=42
    # np.random.seed(seed)
    # tf.random.set_seed(seed)
    # keras.utils.set_random_seed(seed)

    # X_train, y_train, X_val, y_val, X_test, y_test = load_data(seed=42)

    # print(X_train.shape, X_val.shape, X_test.shape)

    # img_rows, img_cols = X_train.shape[1], X_train.shape[2]
    # input_shape = (img_rows, img_cols, 1)
    # print(input_shape)

    # # regression model with two outputs
    # model = build_cnn_multi(input_shape)
    
    # model.compile(loss=[h_numerical_cs_mae, m_numerical_cs_mae],
    #             optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    #             metrics=['accuracy','accuracy'])

    # model.summary()

    # Callbacks
    callbacks = [
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', patience=5, factor=0.5, verbose=1, min_lr=1e-7
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=15, verbose=1, restore_best_weights=True
        )
    ]

    # model.fit(X_train, [y_train[:, 0], y_train[:, 1]],
    #         batch_size=batch_size,
    #         epochs=epochs,
    #         verbose=1,
    #         callbacks=callbacks,
    #         validation_data=(X_val, [y_val[:, 0], y_val[:, 1]]))

    # score = model.evaluate(X_test, [y_test[:, 0], y_test[:, 1]], verbose=0)
    # print('Test loss:', score[0])
    # print('Test hour loss:', score[1])
    # print('Test minute loss:', score[2])
    # print('Test hour accuracy:', score[3])
    # print('Test minute accuracy:', score[4])

    # model.save('saved_models/multi_regression.keras')

    seed=42
    np.random.seed(seed)
    tf.random.set_seed(seed)
    keras.utils.set_random_seed(seed)

    X_train, y_train, X_val, y_val, X_test, y_test = load_data(seed=42)
    print(X_train.shape, X_val.shape, X_test.shape)

    y_train1 = to_categorical(y_train, 12)
    y_val1 = to_categorical(y_val, 12)
    y_test1 = to_categorical(y_test, 12)

    img_rows, img_cols = X_train.shape[1], X_train.shape[2]
    input_shape = (img_rows, img_cols, 1)
    print(input_shape)

    # regression model with two outputs
    model = build_cnn_multi_class_reg(input_shape)

    model.compile(loss=[common_sense_mse_cr, m_numerical_cs_mae],
                optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                metrics=['accuracy','accuracy'])

    model.summary()

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