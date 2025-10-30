import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
from tensorflow.python.framework.ops import SymbolicTensor
from keras.src import ops

from get_dataset import download_data

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

def common_sense_categories_acc(y_true: SymbolicTensor, y_pred: SymbolicTensor) -> SymbolicTensor:
    """
    --------------------------------------------
    Get common sense accuracy for categories
    --------------------------------------------
    :param y_true: tensor filled with tensors with true labels
    :param y_pred: tensor filled with tensors with predicted labels
    :return: common sense accuracy score tensor

    Common sense accuracy metric formula: 
    common sense accuracy = 1-((highest possible common sence loss)/(max possible difference value)) 
    = ((max possible difference value)-(highest possible common sence loss))/(max possible difference value)
    = (amount correct)/(total possible correct amount)
    """
    # read what the class is
    y_true = tf.argmax(y_true, axis=-1) # when printed gives: tf.Tensor(0, shape=(), dtype=int64), here it gives 0 because it is class 0
    y_pred = tf.argmax(y_pred, axis=-1)

    #calc accuracy (tf.cast makes the values have dtype float32 in this case)
    diff = tf.abs(y_true - y_pred)                                # difference (not common sence yet)
    csl = tf.minimum(diff, tf.abs(diff - num_classes))            # highest possible common sence loss (=common sence difference)
    max_diff = tf.cast(tf.math.ceil(num_classes / 2), tf.float32) # max possible difference value
    acc = 1-(tf.cast(csl, tf.float32) / max_diff)                 # accuracy
    return acc


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


def load_data(seed: None) -> tuple[np.ndarray, np.ndarray]:
    url = r"https://surfdrive.surf.nl/index.php/s/Nznt5c48Mzlb2HY/download?path=%2F&files=A1_data_75.zip"
    file = download_data(url)

    url = r"https://surfdrive.surf.nl/index.php/s/Nznt5c48Mzlb2HY/download?path=%2F&files=A1_data_150.zip"
    file = download_data(url)

    X = np.load("data/A1_data_75/images.npy")
    y = np.load("data/A1_data_75/labels.npy")

    X = X / 255
    
    # split data into train, test
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed)
    # split train into train, val
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, random_state=seed)

    return X_train, y_train, X_val, y_val, X_test, y_test


if __name__ == "__main__":
    seed=42
    np.random.seed(seed)
    tf.random.set_seed(seed)
    keras.utils.set_random_seed(seed)

    # preprocessing data
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(seed=42)

    print(X_train.shape, X_val.shape, X_test.shape)

    batch_size = 128
    num_classes = 24
    epochs = 12

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
    model = Sequential([
        Input(shape=input_shape),
        Conv2D(32, kernel_size=(3, 3), activation='relu'),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])


    ################## use own loss and accuracy (and regular accuracy) metric
    model.compile(loss=common_sense_mse,
                optimizer=keras.optimizers.Adadelta(),
                metrics=[common_sense_categories_acc,'accuracy'])


    model.summary()

    model.fit(X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(X_val, y_val))

    score = model.evaluate(X_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    model.save('saved_models/loss_common_sense_mse.keras')


    ################ init new model
    model = Sequential([
        Input(shape=input_shape),
        Conv2D(32, kernel_size=(3, 3), activation='relu'),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    # model.compile(loss=keras.losses.categorical_crossentropy,
    #               optimizer=keras.optimizers.Adadelta(),
    #               metrics=[common_sense_categories_acc,'accuracy'])

    model.compile(loss=keras.losses.MSE,
                optimizer=keras.optimizers.Adadelta(),
                metrics=[common_sense_categories_acc,'accuracy'])

    model.summary()

    model.fit(X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(X_val, y_val))

    score = model.evaluate(X_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test common sense accuracy:', score[1])
    print('Test accuracy:', score[2])

    model.save('saved_models/loss_mse.keras')