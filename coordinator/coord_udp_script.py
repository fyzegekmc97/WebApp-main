import socket
import sys
from threading import Thread
import time
import zipfile
import matplotlib.pyplot as plt
from keras import layers, models, losses
import tensorflow as tf
import numpy as np
import pickle
import communication
import json
import os
from typing import List, Dict
from numpy import mean
from numpy import std
from keras.utils import to_categorical
from typing import Tuple
from math import ceil

numberOfClasses = 2
trainingIterations = 2
totalClients = 1
host = "127.0.0.1"
port = 9000
remote_addresses = [("127.0.0.1", 9001)]

x_test = np.load('images50_percent_testing_data.npy')
y_test = np.load('labels50_percent_testing_data.npy')
x_test /= 255
x_test = tf.convert_to_tensor(x_test, dtype=np.float32)
y_test = tf.convert_to_tensor(y_test, dtype=np.float32)
global loaded_model
print("Image and label data loaded")


class ClientThread(Thread):
    def __init__(self, this_id, this_client, this_address, this_model):
        Thread.__init__(self)
        self.id = this_id
        self.connection = this_client
        self.address = this_address
        self.model = this_model
        self.local_model = []
        self.waiting = True
        self.updated = True
        self.total_time_iteration = 0

    def run(self):
        global total_updates, timeline, iteration_this
        print("Thread for worker with ID ", self.id, " has started running")
        beginning = time.perf_counter()
        while is_training:
            if not self.local_model == 'exit':
                if self.waiting & self.updated:
                    beginning = time.perf_counter()
                    print("Saving model parameters for transmission purposes, targeting client with ID ", self.id)
                    model_json = self.model.to_json()
                    if not os.path.isdir("model_to_send_coordinator" + str(self.id)):
                        os.mkdir("model_to_send_coordinator" + str(self.id))
                    with open("model_to_send_coordinator" + str(self.id) + "/model.json", "w") as json_file:
                        json_file.write(model_json)
                    self.model.save_weights("model_to_send_coordinator" + str(self.id) + "/model.h5")
                    print("Client ", self.id, "Saved model parameters for transmission. Zipping saved model parameters for transmission...")
                    list_of_files_to_zip = ["model_to_send_coordinator" + str(self.id) + "/model.h5", "model_to_send_coordinator" + str(self.id) + "/model.json"]
                    with zipfile.ZipFile("model_to_send_coordinator" + str(self.id) + ".zip", 'w') as zipMe:        
                        for file in list_of_files_to_zip:
                            zipMe.write(file, compress_type=zipfile.ZIP_DEFLATED)
                    print("Client ", self.id, "Zipped model parameters for transmission. Reading zipped file and then transmitting model parameters within a zip file...")
                    some_file = open("model_to_send_coordinator" + str(self.id) + ".zip", "rb")
                    data = some_file.read()
                    send_file_length(self.connection, len(data), self.address)
                    send_file(self.connection, len(data), self.address, data, 1500)
                    print("Client ", self.id, "Zipped model parameters are sent. Going into standby mode for downloading model parameters from the client device...")
                    self.updated = False
                    self.waiting = False
                elif not self.updated:
                    print("Client ", self.id, "In standby mode for receiving model parameters...")
                    file_length = receive_file_size(self.connection, "file_len_")
                    if file_length is None:
                        break
                    my_file_bytes = receive_file(self.connection, file_length, 1500)
                    if my_file_bytes is None:
                        break
                    print("Client ", self.id, " sent data. Attempting to decode received data...")
                    print("Client ", self.id, "Saving received model parameters into a zip file and then unzipping the zip file to a directory...")
                    with open("received_model_coordinator" + str(self.id) +".zip", "wb") as f:
                        f.write(my_file_bytes)
                    with zipfile.ZipFile("received_model_coordinator" + str(self.id) +".zip", 'r') as zip_ref:
                        zip_ref.extractall("received_model_coordinator" + str(self.id))
                    zip_ref.close()
                    print("Client ", self.id, "Model parameters unzipped to directory \"", "received_model_coordinator" + str(self.id), "\"", " Reading model parameters from the mentioned directory...")
                    json_file = open("received_model_coordinator" + str(self.id) + "/model_to_send_worker" + str(self.id) + "/model.json", "r")
                    model_json = json_file.read()
                    json_file.close()
                    loaded_model = models.model_from_json(model_json)
                    loaded_model.load_weights("received_model_coordinator" + str(self.id) + "/model_to_send_worker" + str(self.id) + "/model.h5")
                    print("Client ", self.id, "Successfully uploaded model parameters.")
                    self.local_model = loaded_model
                    total_updates += 1
                    self.updated = True
                    self.total_time_iteration = time.perf_counter() - beginning
                    print("Iteration took: ", self.total_time_iteration, "seconds")
            else:
                break
        print("Client ", self.id, " is disconnected.")
        self.connection.sendto("exit".encode(), self.address)
        self.updated = False
        self.connection.close()
        print("Quitting thread for client with ID ", self.id)


def receive_file(some_socket:socket.socket, file_length: int, packet_size: int = 1500):
    bytes_buffer = bytearray()
    while len(bytes_buffer) < file_length:
        recvd_bytes = some_socket.recv(packet_size)
        try:
            decoded = recvd_bytes.decode()
            if "exit" in decoded:
                return bytes_buffer
        except:
            pass
        bytes_buffer.extend(recvd_bytes)
        print("Received bytes of length: ", len(recvd_bytes), " completed receiving ", len(bytes_buffer), "/", file_length)
    return bytes_buffer


def receive_ID(some_socket: socket.socket, ID_keyword: str):
    recvd_bytes = some_socket.recv(1500)
    try:
        decoded = recvd_bytes.decode()
        if "exit" in decoded:
            return None
    except:
        pass
    if ID_keyword in recvd_bytes.decode():
        return int(recvd_bytes.decode().replace(ID_keyword, ""))
    else:
        print("Did not receive ID")
        sys.exit(-1)


def model_average(clients_list: List[ClientThread], central_model: Dict[str, list]):
    temp_model = {"weights": []}
    if type(clients_list[0].local_model) == dict:
        for i in range(len(clients_list[0].local_model["weights"])):
            temp_model['weights'].append(clients_list[0].local_model["weights"][i])
        for i in range(1,len(clients_list)):
            for k in range(len(temp_model["weights"])):
                temp_model["weights"][k].assign_add(clients_list[i].local_model["weights"][k])
        for i in range(len(temp_model["weights"])):
            temp_model["weights"][i].assign(temp_model["weights"][i] / totalClients)
    elif type(clients_list[0].local_model) == models.Sequential:
        for i in range(len(clients_list[0].local_model.weights)):
            temp_model['weights'].append(clients_list[0].local_model.weights[i])
        for i in range(1,len(clients_list)):
            for k in range(len(temp_model["weights"])):
                temp_model["weights"][k].assign_add(clients_list[i].local_model.weights[k])
        for i in range(len(temp_model["weights"])):
            temp_model["weights"][i].assign(temp_model["weights"][i] / totalClients)
    central_model = temp_model
    model.set_weights(central_model["weights"])


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


def readmit_clients(client_list):
    return client_list

clients_number = 0
clients_list = []
total_updates = 0
is_training = True
iteration_this = 0
accuracy_test = np.zeros([trainingIterations])
central_model = {'weights': []}

model = models.Sequential()
model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=(200, 200, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(2))
model.summary()
model.compile(optimizer='adam', loss=losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
history = model.fit(x_test, y_test, epochs=1)
ServerSideSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
ServerSideSocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
for i in range(len(model.weights)):
    central_model['weights'].append(tf.Variable(model.weights[i]))

print("Global parameters initialized.")

try:
    ServerSideSocket.bind((host, port))
except socket.error as e:
    print(str(e))
print("Socket bound")

while clients_number < totalClients:
    send_ID(ServerSideSocket,clients_number, remote_addresses[clients_number], "ID_")
    print("Sent ID number to: ", remote_addresses[clients_number])
    client = ClientThread(clients_number, ServerSideSocket, remote_addresses[clients_number], model)
    clients_list.append(client)
    clients_number += 1
    time.sleep(5)

print("All clients created.")

print("Training started...")
for client_this in clients_list:
    client_this.start()


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


while is_training:
    if not total_updates < totalClients:
        print("All models received from clients. Averaging model parameters...")
        model_average(clients_list, central_model)
        print("Averaged model parameters to obtain the global model. Assigning the global to each client for later transmission...")
        for client_this in clients_list:
            client_this.model = model
            client_this.waiting = True
        print("Assigned global model parameters to the clients.")
        iteration_this += 1
        is_training = iteration_this < trainingIterations
        print(iteration_this)
        total_updates = 0


print("Training ended.")
# waiting for client communication to finish
for client_this in clients_list:
    client_this.join()

# close the port
ServerSideSocket.close()
score = model.evaluate(x=x_test, y=y_test, verbose=1, steps=30)
print("Final %s: %.2f%%" % (model.metrics_names[1], score[1]*100))