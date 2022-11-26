from genericpath import isdir
import socket
import pickle
import sys
import zipfile
import zlib
import communication
import numpy as np
import tensorflow as tf
import time
from keras import models, losses, metrics, layers
import os
from typing import Any
import random

host = "192.168.1.45"
port = 3001
number_of_epochs = 20

x_train = np.load('images50_percent_training_data.npy')
y_train = np.load('labels50_percent_training_data.npy')
x_train = x_train / 255

def mini_batch(input_images: np.ndarray, image_labels: np.ndarray, batch_size: int):
    zip_obj = zip(input_images, image_labels)
    samples = random.sample(list(zip_obj), batch_size)
    output_images = np.ndarray(shape=(1, input_images.shape[1], input_images.shape[2]))
    output_labels = np.ndarray(shape=(1, image_labels.shape[1]))
    temp_image = np.ndarray(shape=(1, input_images.shape[1], input_images.shape[2]))
    temp_label = np.ndarray(shape=(1, image_labels.shape[1]))
    output_images[0] = samples[0][0]
    output_labels[0] = samples[0][1]
    for i in range(1, len(samples)):
        temp_image[0] = samples[i][0]
        temp_label[0] = samples[i][1]
        output_images = np.concatenate([output_images, temp_image])
        output_labels = np.concatenate([output_labels, temp_label])
    return output_images, output_labels

print("Image data and label data is loaded")
clientSocket = socket.socket()

print('Waiting for connection response')
while True:
    try:
        clientSocket.connect((host, port))
        break
    except socket.error as e:
        continue
print('Connected')
message = communication.receive_data(clientSocket)
my_id = pickle.loads(message)
if type(my_id) is not int:
    print("ID was not obtained successfully")
    clientSocket.close()
    sys.exit(-1)
else:
    print("ID: ", my_id)
    print("----------")

if not os.path.isdir("received_model_worker" + str(my_id) + "/"):
    os.mkdir("received_model_worker" + str(my_id))

if not os.path.isdir("model_to_send_worker" + str(my_id) + "/"):
    os.mkdir("model_to_send_worker" + str(my_id))

print("Directories created for worker with ID ", my_id)
curr_iter = 0
global loaded_model
while True:
    print("Iteration ", curr_iter, " has started.")
    print("Receiving model parameters from the client socket...")
    begin_time_worker = time.perf_counter()
    message = communication.receive_data(clientSocket)
    print("Model parameters downloaded")
    try:
        print("Attempting to unpickle downloaded data")
        message_decoded = pickle.loads(message)
        print("Successfully unpickled data")
        if message_decoded == "exit":
            print("Exiting training session")
            break
    except:
        print("Data was not appropriate for unpickling")
        message_decoded = ""
        pass
    try:
        print("Attempting to zip received model files")
        with open("received_model_worker" + str(my_id) + ".zip", "wb") as f:
            f.write(message)
        with zipfile.ZipFile("received_model_worker" + str(my_id) +".zip", 'r') as zip_ref:
            zip_ref.extractall("received_model_worker"+ str(my_id))
        zip_ref.close()
        print("Zipped received model files")
        print("Attempting to read model files from zip file...")
        json_file = open("received_model_worker" + str(my_id) + "/model_to_send_coordinator" + str(my_id) + "/model.json", "r")
        model_json = json_file.read()
        json_file.close()
        loaded_model = models.model_from_json(model_json)
        loaded_model.load_weights("received_model_worker" + str(my_id) + "/model_to_send_coordinator" + str(my_id) + "/model.h5")
        print("Model weights loaded.")
        loaded_model.compile(optimizer="adam", loss=losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])
        print("Model files and model weights loaded successfully")
        print("Training on loaded model...")
        mb_size = 32
        for i in range(number_of_epochs):
            print("Epoch ", str(i+1))
            images, labels = mini_batch(x_train, y_train, mb_size)
            print(mb_size, " images chosen at random")
            loaded_model.train_on_batch(x=images, y=labels)
            print("Epoch ", str(i+1), "finished.")
        print("Training has finished")
        print("Writing new model parameters to zip files.")
        model_json = loaded_model.to_json()
        with open("model_to_send_worker" + str(my_id) + "/model.json", "w") as json_file:
            json_file.write(model_json)
        loaded_model.save_weights("model_to_send_worker" + str(my_id) + "/model.h5")
        print("Saved new model to disk")
        print("Zipping saved model parameters for transmiossion")
        list_of_files_to_zip = ["model_to_send_worker" + str(my_id) + "/model.h5", "model_to_send_worker" + str(my_id) + "/model.json"]
        with zipfile.ZipFile("model_to_send_worker" + str(my_id) + ".zip", 'w') as zipMe:        
            for file in list_of_files_to_zip:
                zipMe.write(file, compress_type=zipfile.ZIP_DEFLATED)
        print("Zipped new model parameters for transmission. Reading and then transferring the relevant zip file...")
        some_file = open("model_to_send_worker" + str(my_id) + ".zip", "rb")
        my_bytes = some_file.read()
        communication.send_data(clientSocket, my_bytes)
        print("Read and transmitted the relevant zip file.")
        total_time_worker = time.perf_counter() - begin_time_worker
        print("Iteration took a total time of ", total_time_worker, "seconds of time")
        curr_iter += 1
    except:
        pass
data = pickle.dumps("exit")
communication.send_data(clientSocket, data)
print("Exit data sent")
time.sleep(1)
clientSocket.close()
print('Connection closed.')
