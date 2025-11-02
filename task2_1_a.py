import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
from tensorflow.python.framework.ops import SymbolicTensor
from keras.src import ops
import os

from get_dataset import download_data

num_classes = 24
batch_size = 128
epochs = 50

def print_metrics(pred_hours, true_hours, model_name, num_classes=24):
    """Print comprehensive evaluation metrics."""
    diff_min = common_sense_categories_loss(true_hours,pred_hours)
    
    mean_err = np.mean(diff_min)
    median_err = np.median(diff_min)
    std_err = np.std(diff_min)
    max_err = np.max(diff_min)
    
    within_30 = np.mean(diff_min <= 1) * 100

    minutes_per_class = 60//(num_classes//12)

    print(f"\n{'=' * 80}")
    print(f"{model_name} - TEST SET RESULTS")
    print(f"{'=' * 80}")
    print(f"Mean Absolute Error:    {mean_err:.2f} of {minutes_per_class} minutes")
    print(f"Median Absolute Error:  {median_err:.2f} of {minutes_per_class} minutes")
    print(f"Std Deviation:          {std_err:.2f} of {minutes_per_class} minutes")
    print(f"Max Error:              {max_err:.2f} of {minutes_per_class} minutes")
    print(f"\nAccuracy within thresholds:")
    print(f"  Within 30 minutes:    {within_30:.1f}%")
    
    return {
        'mean': mean_err,
        'median': median_err,
        'std': std_err,
        'max': max_err,
        'within_30': within_30,
        'predictions': pred_hours,
        'errors': diff_min
    }

def to_categorical(y, num_classes):
    """
    Convert time to categorical labels.

    :param y: numpy array of shape (num_samples, 2) where the first column is hour (0-11)
              and the second column is minute (0-59)
    :param num_classes: total number of classes
    :return: numpy array with one-hot encoded labels that represent the class that the elements of y belong to
    """
    # check if it is already one-hot encoded
    if y.shape[1] == num_classes:
        return y

    class_ph = num_classes//12                    # amount of classes per hour
    class_pm = 60//class_ph                       # amount of classes per minute
    class_ = y[:, 0]*class_ph + y[:, 1]//class_pm # if num_class is 24, then per 30 min is the next class.

    return keras.utils.to_categorical(class_, num_classes)

@tf.keras.utils.register_keras_serializable()
def common_sense_categories_loss(y_true: SymbolicTensor, y_pred: SymbolicTensor) -> SymbolicTensor:
    """
    --------------------------------------------
    Get common sense loss for categories
    --------------------------------------------
    :param y_true: tensor filled with tensors with true labels
    :param y_pred: tensor filled with tensors with predicted labels
    :return: accuracy loss tensor

    Common sense loss formula: 
    (highest possible common sence loss) = cls = min(|true_class-pred_class|, ||true_class-pred_class| - number_of_classes|)
    """
    # read what the class is
    y_true = tf.argmax(y_true, axis=-1) # when printed gives: tf.Tensor(0, shape=(), dtype=int64), here it gives 0 because it is class 0
    y_pred = tf.argmax(y_pred, axis=-1)

    #calc common sense loss (cls)
    diff = tf.abs(y_true - y_pred)                     # difference (not common sence yet)
    csl = tf.minimum(diff, tf.abs(diff - num_classes)) # highest possible common sence loss (=common sence difference)
    return csl                                         # return common sense loss


@tf.keras.utils.register_keras_serializable()
def common_sense_mse(y_true,y_pred):
    """
    --------------------------------------------
    Get mean squared error with common sense
    --------------------------------------------
    :param y_true: tensor filled with tensors with true labels
    :param y_pred: tensor filled with tensors with predicted labels
    :return: mean squared error (MSE) with common sense incorporated (MSE, but we make classes that are further away have more loss)

    Make a distribution that is bigger in value when farther away from the correct class. 
    That distribution is then used as weights to alter the mean squared error (MSE).
    What is returned is: MSE * distribution
    """
    # precausion, make them have the same dtype
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # get squared difference
    diff = tf.square(y_true - y_pred)  # shape (batch, num_classes)

    # get the class
    class_by_mult = tf.range(num_classes, dtype=tf.float32)
    true_class = tf.reduce_sum(y_true * class_by_mult, axis=-1) 
    true_class = tf.cast(true_class, tf.int32)  # must be int for tf.roll
    
    # get distribution based on absolute distance for when index = 0
    dist = [1]
    max_amount = np.ceil(num_classes / 2)
    for i in range(2,int(max_amount)+1):
        dist.append(i)
    for i in range(int(max_amount)+1,1,-1):
        dist.append(i)
    dist = tf.constant(dist, dtype=tf.float32)

    # rotate distribution so that 1 aligns with true class
    # Ex: when index=0 is the correct class, then the distribution 
    # for 4 classes would look like this: [1,2,3,2]
    # , when index = 1 it would look like this: [2,1,2,3]
    def rotate_dist(idx):
        idx = tf.reshape(idx, ())  # make is a scalar
        row = tf.roll(dist, shift=idx, axis=0)
        return row

    dist_matrix = tf.map_fn(rotate_dist, true_class, fn_output_signature=tf.float32) # loop over true_class rows

    # multiply squared difference by distance weights
    diff_with_common_sense = diff * dist_matrix

    # average over classes to get the MSE with common_sense incorporated
    loss = tf.reduce_mean(diff_with_common_sense, axis=-1)
    return loss


def load_data(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    url = r"https://surfdrive.surf.nl/index.php/s/Nznt5c48Mzlb2HY/download?path=%2F&files=A1_data_75.zip"
    download_data(url)

    url = r"https://surfdrive.surf.nl/index.php/s/Nznt5c48Mzlb2HY/download?path=%2F&files=A1_data_150.zip"
    download_data(url)

    X = np.load("data/A1_data_75/images.npy")
    y = np.load("data/A1_data_75/labels.npy")

    X = X / 255
    
    # split data into train, test
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed)
    # split train into train, val
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, random_state=seed)

    return X_train, y_train, X_val, y_val, X_test, y_test


def build_cnn_catagorical(input_shape, num_classes):
    """CNN for classification (predicting classes)."""
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

    outputs = Dense(num_classes, activation='softmax')(x)

    return keras.Model(inputs, outputs, name="cnn_classification")

if __name__ == "__main__":
    os.makedirs('saved_models', exist_ok=True)

    seed=42
    np.random.seed(seed)
    tf.random.set_seed(seed)
    keras.utils.set_random_seed(seed)

    # preprocessing data
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(seed=42)

    print(X_train.shape, X_val.shape, X_test.shape)

    img_rows, img_cols = X_train.shape[1], X_train.shape[2]
    input_shape = (img_rows, img_cols, 1)

    if K.image_data_format() == 'channels_first':
        X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
        X_val = X_val.reshape(X_val.shape[0], 1, img_rows, img_cols)
        X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
        X_val = X_val.reshape(X_val.shape[0], img_rows, img_cols, 1)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    print(X_train.shape, X_val.shape, X_test.shape)

    # Convert labels to one-hot encoding
    # ex.
    # y_train = to_categorical(y_train, 720) # when doing 720 labels use this

    y_train = to_categorical(y_train, num_classes)
    print(f"Class and amount:\n{np.sum(y_train,axis=0)}\n") # print class distribution, the index is the class, the amount is how many times that class is in the data

    y_val = to_categorical(y_val, num_classes)
    y_test = to_categorical(y_test, num_classes)

    print(y_train.shape, y_val.shape, y_test.shape)

    # ####################################### make model
    model = build_cnn_catagorical(input_shape, num_classes)
    

    ################## use own loss and accuracy (and regular accuracy) metric
    model.compile(loss=common_sense_mse,
                optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                metrics=[common_sense_categories_loss,'accuracy'])


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

    model.fit(X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            callbacks=callbacks,
            validation_data=(X_val, y_val))

    # Load best model
    model = keras.models.load_model('saved_models/temp_best.keras')
    
    # Evaluate the model
    pred_categorical = model.predict(X_test,verbose=0)

    metrics_categorical = print_metrics(pred_categorical, y_test, model_name="common_sense_mse")

    score = model.evaluate(X_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test common sense loss:', score[1])
    print('Test accuracy:', score[2])

    model.save('saved_models/loss_common_sense_mse.keras')


    ################ init new model
    model = build_cnn_catagorical(input_shape, num_classes)

    model.compile(loss=keras.losses.MSE,
                optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                metrics=[common_sense_categories_loss,'accuracy'])


    model.summary()

    model.fit(X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            callbacks=callbacks,
            validation_data=(X_val, y_val))
    
    # Load best model
    model = keras.models.load_model('saved_models/temp_best.keras')

    score = model.evaluate(X_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test common sense loss:', score[1])
    print('Test accuracy:', score[2])

    model.save('saved_models/loss_mse.keras')

    ################ init new model
    model = build_cnn_catagorical(input_shape, num_classes)

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                  metrics=[common_sense_categories_loss,'accuracy'])

    model.summary()

    model.fit(X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            callbacks=callbacks,
            validation_data=(X_val, y_val))

    # Load best model
    model = keras.models.load_model('saved_models/temp_best.keras')

    score = model.evaluate(X_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test common sense loss:', score[1])
    print('Test accuracy:', score[2])

    model.save('saved_models/loss_crossentropy.keras')