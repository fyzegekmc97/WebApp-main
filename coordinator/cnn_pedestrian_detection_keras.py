import numpy as np
import pandas as pd
import os
from xml.etree import ElementTree
from matplotlib import pyplot as plt
import tensorflow as tf
from keras import datasets, layers, models, callbacks
from numpy.random import seed
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau
import os
from xml.etree import ElementTree
import tensorflow_model_optimization as tf_mod_opt
import tensorflow.lite as tfl
import datetime

seed(1)

n_classes = 2
size = (200, 200)
x_train_new = np.load("images60_percent_training_data.npy")
y_train = np.load("labels60_percent_training_data.npy")

x_test_new = np.load("images40_percent_testing_data.npy")
y_test = np.load("labels40_percent_testing_data.npy")

pruning_schedule = tf_mod_opt.sparsity.keras.PolynomialDecay(initial_sparsity=0.0, final_sparsity=0.5, begin_step=2000, end_step=4000)

model = models.Sequential()
model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=(200, 200, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(2))
model.summary()
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

preset_batch_size = 50

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1, save_freq=5*preset_batch_size)

history = model.fit(x_train_new, y_train, batch_size=preset_batch_size, epochs=15, validation_data=(x_test_new, y_test), callbacks=[cp_callback, tensorboard_callback])

latest = tf.train.latest_checkpoint(checkpoint_dir)
print(latest)

model.save("my_model")

models.save_model(model, "my_model.h5", True, False, 'h5', None)

loaded_model = models.load_model("my_model")

print(type(model))

""" model_for_pruning = tf_mod_opt.sparsity.keras.prune_low_magnitude(loaded_model, pruning_schedule=pruning_schedule)

model_for_pruning.fit() """

converter = tfl.TFLiteConverter.from_saved_model("my_model")
tf_lite_model = converter.convert()

with open('model.tflite', 'wb') as f:
  f.write(tf_lite_model)

model_from_checkpoints = models.Sequential()
model_from_checkpoints.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=(200, 200, 1)))
model_from_checkpoints.add(layers.MaxPooling2D((2, 2)))
model_from_checkpoints.add(layers.Conv2D(32, (3, 3), activation='relu'))
model_from_checkpoints.add(layers.MaxPooling2D((2, 2)))
model_from_checkpoints.add(layers.Flatten())
model_from_checkpoints.add(layers.Dense(128, activation='relu'))
model_from_checkpoints.add(layers.Dense(2))
model_from_checkpoints.summary()
model_from_checkpoints.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

model_from_checkpoints.load_weights(latest)

history_from_checkpoints = model.fit(x_train_new, y_train, batch_size=preset_batch_size, epochs=15, validation_data=(x_test_new, y_test))

print(history_from_checkpoints)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training', 'Test'], loc='lower right')
plt.savefig("60_40_split_accuracy_results.eps", dpi = 600, format="eps")

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training', 'Test'], loc='upper right')
plt.savefig("60_40_split_loss_results.eps", dpi = 600, format="eps")

plt.clf()
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training', 'Test'], loc='upper left')

plt.subplot(1,2,2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training', 'Test'], loc='upper left')

plt.savefig("60_40_split_loss_accuracy_results.eps", dpi = 600, format="eps")