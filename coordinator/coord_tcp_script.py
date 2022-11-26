import socket
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

numberOfClasses = 2
trainingIterations = 20
totalClients = 1
host = "127.0.0.1"
port = 3001

x_test = np.load('images50_percent_testing_data.npy')
y_test = np.load('labels50_percent_testing_data.npy')
x_test /= 255
x_test = tf.convert_to_tensor(x_test, dtype=np.float32)
y_test = tf.convert_to_tensor(y_test, dtype=np.float32)
global loaded_model
print("Image and label data loaded")

clients_number = 0
clients_list = []
total_updates = 0
is_training = True
iteration_this = 0
accuracy_test = np.zeros([trainingIterations])
central_model = {'weights': []}

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
        self.compute_latencies = []
        self.communication_latencies = []

    def run(self):
        global total_updates, timeline, iteration_this
        print("Thread for worker with ID ", self.id, " has started running")
        while True:
            if not self.local_model == 'exit':
                if self.waiting & self.updated:
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
                    communication.send_data(self.connection, data)
                    print("Client ", self.id, "Zipped model parameters are sent. Going into standby mode for downloading model parameters from the client device...")
                    self.updated = False
                    self.waiting = False
                elif not self.updated:
                    print("Client ", self.id, "In standby mode for receiving model parameters...")
                    message = communication.receive_data(self.connection)
                    print("Client ", self.id, " sent data. Attempting to decode received data...")
                    try:
                        message_decoded = pickle.loads(message)
                        print("Unpickled data.")
                        if message_decoded == "exit":
                            print("Exiting session...")
                            break
                    except:
                        message_decoded = ""
                        pass
                    try:
                        print("Client ", self.id, "Saving received model parameters into a zip file and then unzipping the zip file to a directory...")
                        with open("received_model_coordinator" + str(self.id) +".zip", "wb") as f:
                            f.write(message)
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
                    except:
                        pass
                    if not data:
                        break
                    self.local_model = loaded_model
                    total_updates += 1
                    self.updated = True
            else:
                break
        print("Client ", self.id, " is disconnected.")
        self.connection.close()

def readmit_clients(client_list):
    return client_list

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
ServerSideSocket = socket.socket()
ServerSideSocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
for i in range(len(model.weights)):
    central_model['weights'].append(tf.Variable(model.weights[i]))

print("Global parametrers initialized.")

try:
    ServerSideSocket.bind((host, port))
except socket.error as e:
    print(str(e))
print(host + ':' + str(port) + ' is listening..')
ServerSideSocket.listen(5)

while clients_number < totalClients:
    Client, Address = ServerSideSocket.accept()
    print('Connected to: ' + Address[0] + ':' + str(Address[1]))
    message_identity = clients_number
    communication.send_data(Client, pickle.dumps(message_identity))
    client = ClientThread(clients_number, Client, Address, model)
    clients_list.append(client)
    clients_number += 1

print("All clients created.")

print("Training started...")
for client_this in clients_list:
    client_this.start()

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

while total_updates < totalClients:
    time.sleep(.1)

for client_this in clients_list:
    communication.send_data(client_this.connection, pickle.dumps("exit"))
    client_this.updated = False

print("Training ended.")
# waiting for client communication to finish
for client_this in clients_list:
    client_this.join()

# close the port
ServerSideSocket.close()
score = model.evaluate(x=x_test, y=y_test, verbose=1, steps=30)
print("Final %s: %.2f%%" % (model.metrics_names[1], score[1]*100))