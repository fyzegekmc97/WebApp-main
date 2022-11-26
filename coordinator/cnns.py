import numpy as np
from keras.optimizers import SGD
from keras.datasets import cifar10
from keras.models import Sequential
from keras.utils import np_utils as utils
from keras.layers import Dropout, Dense, Flatten, Conv1D
from keras.layers.convolutional import Conv2D, MaxPooling2D, MaxPooling1D
from keras.callbacks import EarlyStopping

(X, y), (X_test, y_test) = cifar10.load_data()
print(type(X))
print(type(X_test))
print(type(y))
print(type(y_test))
X, X_test = X.astype("float32") / 255.0, X_test.astype("float32") / 255.0
y, y_test = utils.to_categorical(y, 10), utils.to_categorical(y_test, 10)
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(32, 32, 3), padding="same", activation="relu"))
model.add(Dropout(0.2))
model.add(Conv2D(32, (3, 3), activation="relu", padding="valid"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(10, activation="softmax"))
model.compile(
    loss="categorical_crossentropy",
    optimizer=SGD(momentum=0.5, decay=0.0004),
    metrics=["accuracy"],
)
model.fit(X, y, validation_data=(X_test, y_test), epochs=25, batch_size=512)
accuracy = model.evaluate(X_test, y_test)[1]*100
print("Accuracy: ", accuracy)
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=32, kernel_size=3, padding="same", activation="relu"))
model.add(Conv1D(filters=32, kernel_size=3, padding="same", activation="relu"))
model.add(Dropout(0.2))
earlystop = EarlyStopping(
    monitor="val_loss", min_delta=0, patience=3, verbose=1, restore_best_weights=True
)
