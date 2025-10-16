import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
from tensorflow.python.framework.ops import SymbolicTensor
from keras.src import ops

from get_dataset import download_data

np.random.seed(42)

url = r"https://surfdrive.surf.nl/index.php/s/Nznt5c48Mzlb2HY/download?path=%2F&files=A1_data_75.zip"
file = download_data(url)

url = r"https://surfdrive.surf.nl/index.php/s/Nznt5c48Mzlb2HY/download?path=%2F&files=A1_data_150.zip"
file = download_data(url)

X = np.load("data/A1_data_75/images.npy")
y = np.load("data/A1_data_75/labels.npy")

X = X / 255

print(X.shape)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, random_state=42)

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

def to_categorical(y, num_classes):
    """
    Convert time to categorical labels.

    :param y: numpy array of shape (num_samples, 2) where the first column is hour (0-11)
              and the second column is minute (0-59)
    :param num_classes: total number of classes
    :return: numpy array with one-hot encoded labels
    """
    # check if it is already one-hot encoded
    if y.shape[1] == num_classes:
        return y

    class_ = y[:, 0]*2 + y[:, 1]//30

    return keras.utils.to_categorical(class_, num_classes)

# Convert labels to one-hot encoding
y_train = to_categorical(y_train, num_classes)
y_val = to_categorical(y_val, num_classes)
y_test = to_categorical(y_test, num_classes)

print(y_train.shape, y_val.shape, y_test.shape)

def common_sense_categories(y_true: SymbolicTensor, y_pred: SymbolicTensor) -> SymbolicTensor:
    """
    Common sense accuracy metric: common sense accuracy = 1-(highest possible common sence loss)/(max possible difference value)

    :param y_true: true labels
    :param y_pred: predicted labels
    :return: accuracy score
    """
    # y_true_np = ops.convert_to_numpy(y_true)
    # y_pred_np = ops.convert_to_numpy(y_pred)
    # print(y_true_np, y_pred_np)
    # TODO: CONVERT TO NUMPY AND BACK
    true_class = np.argmax(y_true)
    pred_class = np.argmax(y_pred)
    
    acc = 1-(min(abs(true_class - pred_class), abs(abs(true_class - pred_class) - num_classes))/np.ceil(num_classes/2))
    return acc

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

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.summary()

model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_val, y_val))

score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

