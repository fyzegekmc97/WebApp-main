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
from typing import Tuple
from math import ceil


number_of_epochs = 10
iter_len = 2
remote_address = ("127.0.0.1", 9000)
local_address = ("127.0.0.1", 9001)
x_train = np.load('images50_percent_training_data.npy')
y_train = np.load('labels50_percent_training_data.npy')
x_train = x_train / 255


def receive_file(some_socket:socket.socket, file_length: int, packet_size: int = 1500):
    bytes_buffer = bytearray()
    while len(bytes_buffer) < file_length:
        recvd_bytes = some_socket.recv(packet_size)
        try:
            decoded = recvd_bytes.decode()
            if "exit" in decoded:
                return None
        except:
            pass
        bytes_buffer.extend(recvd_bytes)
        print("Received bytes of length: ", len(recvd_bytes), " completed receiving ", len(bytes_buffer), "/", file_length)
    return bytes_buffer


def receive_ID(some_socket: socket.socket, ID_keyword: str):
    recvd_bytes = some_socket.recv(1500)
    if ID_keyword in recvd_bytes.decode():
        return int(recvd_bytes.decode().replace(ID_keyword, ""))
    elif "exit" in recvd_bytes.decode():
        return None
    else:
        print("Did not receive ID")
        sys.exit(-1)


def send_file_length(some_socket: socket.socket, data_length: int, remote_address: Tuple[str,int]):
    some_socket.sendto(("file_len_" + str(data_length)).encode(), remote_address)


def send_file(some_socket: socket.socket, data_length: int, remote_address: Tuple[str,int], data: bytes, packet_size: int = 1500):
    total_packet_count = ceil(data_length / packet_size)
    for i in range(0,total_packet_count):
        if i != total_packet_count - 1:
            some_socket.sendto(data[packet_size*i:i*packet_size+packet_size], remote_address)
        else:
            some_socket.sendto(data[packet_size*i:], remote_address)
        time.sleep(0.1)


def send_ID(some_socket: socket.socket, ID_number: int, remote_address: Tuple[str,int], ID_keyword: str):
    some_socket.sendto((str(ID_number) + ID_keyword).encode(), remote_address)


def receive_file_size(some_socket: socket.socket, file_length_keyword: str):
    while True:
        print("Started receival")
        try:
            bytez = some_socket.recv(1500)
            try:
                decoded_bytes = bytez.decode()
            except:
                continue
            print(decoded_bytes)
            if file_length_keyword in decoded_bytes:
                print("Prefix found")
                print(type(decoded_bytes))
                file_len = int(decoded_bytes.replace(file_length_keyword, ""))
                print(file_len)
                return file_len
            elif "exit" in decoded_bytes:
                return None
            else:
                print("Did not get file size but got: ", decoded_bytes)
                continue
        except:
            pass


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
clientSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
clientSocket.bind(local_address)
print('Client socket bound')
my_id = receive_ID(clientSocket, "ID_")
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
loaded_model = models.Sequential()
loaded_model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=(200, 200, 1)))
loaded_model.add(layers.MaxPooling2D((2, 2)))
loaded_model.add(layers.Conv2D(32, (3, 3), activation='relu'))
loaded_model.add(layers.MaxPooling2D((2, 2)))
loaded_model.add(layers.Flatten())
loaded_model.add(layers.Dense(128, activation='relu'))
loaded_model.add(layers.Dense(2))
loaded_model.compile(optimizer='adam', loss=losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

while curr_iter < iter_len:
    iter_begin = time.perf_counter()
    print("Iteration ", curr_iter, " has started.")
    print("Receiving model parameters from the client socket...")
    file_length = receive_file_size(clientSocket, "file_len_")
    if file_length is None:
        break
    data = receive_file(clientSocket, file_length, 1500)
    print("Model parameters downloaded")
    print("Attempting to zip received model files")
    with open("received_model_worker" + str(my_id) + ".zip", "wb") as f:
        f.write(data)
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
    try:
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
        send_file_length(clientSocket, len(my_bytes), remote_address)
        send_file(clientSocket, len(my_bytes), remote_address, my_bytes, 1500)
        print("Read and transmitted the relevant zip file.")
        curr_iter += 1
    except:
        pass
    iter_total_time = time.perf_counter() - iter_begin
    print("Iteration took: ", iter_total_time, "seconds")

data = "exit".encode()
clientSocket.sendto(data, remote_address)
print("Exit data sent")
time.sleep(1)
clientSocket.close()
print('Connection closed.')
