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
from task2_1_a import load_data
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

batch_size = 128
epochs = 50

def split_to_diff_min(pred_hours, true_hours):
    """Calculate absolute difference in minutes between predicted and true times."""
    pred_total_min = (pred_hours[0] * 60) + pred_hours[1]
    true_total_min = (true_hours[0] * 60) + true_hours[1]

    diff_min = np.abs(pred_total_min - true_total_min)

    # Handle wrap-around
    wrap_around_diff = 720 - diff_min
    diff_min = np.minimum(diff_min, wrap_around_diff)
    
    return diff_min

def print_metrics(pred_hours, true_hours, model_name):
    """Print comprehensive evaluation metrics."""
    diff_min = split_to_diff_min(pred_hours, true_hours)
    
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
        'predictions': pred_hours,
        'errors': diff_min
    }

@tf.keras.utils.register_keras_serializable()
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

@tf.keras.utils.register_keras_serializable()
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

def build_cnn_multi(input_shape):
    """CNN with multi-headed regression (two outputs for hours and minutes)."""
    inputs = keras.Input(shape=input_shape)
    
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)
    
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)
    
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)
    
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)

    hour_output = Dense(1, activation='linear')(x)
    minute_output = Dense(1, activation='linear')(x)

    return keras.Model(inputs, [hour_output, minute_output], name="cnn_classification")


if __name__ == "__main__":
    os.makedirs('saved_models', exist_ok=True)
    
    seed=42
    np.random.seed(seed)
    tf.random.set_seed(seed)
    keras.utils.set_random_seed(seed)

    X_train, y_train, X_val, y_val, X_test, y_test = load_data(seed=42)

    print(X_train.shape, X_val.shape, X_test.shape)

    # # scale y_train to [0, 1]
    # print(y_train)
    # scaler = MinMaxScaler(feature_range=(0, 1))
    # scaler.fit(y_train)
    # y_train = scaler.transform(y_train)
    # y_val = scaler.transform(y_val)
    # y_test = scaler.transform(y_test)
    # print(y_train)

    img_rows, img_cols = X_train.shape[1], X_train.shape[2]
    input_shape = (img_rows, img_cols, 1)
    print(input_shape)

    # regression model with two outputs
    model = build_cnn_multi(input_shape)
    
    model.compile(loss=[h_numerical_cs_mae, m_numerical_cs_mae],
                optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                metrics=['accuracy','accuracy'])

    model.summary()

    # Callbacks
    callbacks = [
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', patience=5, factor=0.5, verbose=1, min_lr=1e-7
        ),
        keras.callbacks.ModelCheckpoint(
            'saved_models/temp_best.keras', 
            monitor='val_loss', 
            save_best_only=True, 
            verbose=1
        )
    ]

    model.fit(X_train, [y_train[:, 0], y_train[:, 1]],
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            callbacks=callbacks,
            validation_data=(X_val, [y_val[:, 0], y_val[:, 1]]))
    
    # Load best model
    model = keras.models.load_model('saved_models/temp_best.keras')

    score = model.evaluate(X_test, [y_test[:, 0], y_test[:, 1]], verbose=0)
    print('Test loss:', score[0])
    print('Test hour loss:', score[1])
    print('Test minute loss:', score[2])
    print('Test hour accuracy:', score[3])
    print('Test minute accuracy:', score[4])

    model.save('saved_models/multi_regression.keras')