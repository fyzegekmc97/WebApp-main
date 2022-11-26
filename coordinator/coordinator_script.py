import socket
from threading import Thread
import time
import textwrap
import matplotlib.pyplot as plt
import sys
try:
    import tensorflow as tf
except KeyboardInterrupt:
    sys.exit(0)
import numpy as np
import math
import pickle
import json
from receive_tcp_send_udp_coordinator import *
from receive_udp_send_tcp_coordinator import *
from typing import Any
from getmac import get_mac_address
import random


# Global parameters needed for script:
numberOfLayers = 3  # Total number of layers expected within the neural network
activationFunction = ['relu', 'relu', 'softmax']  # Activation functions of each layer
numberOfNeurons = [784, 128, 64, 2]  # length of this array must be numberOfLayers + 1
fullyConnected = ['yes', 'yes', 'yes']  # length of this array must be numberofLayers + 1
numberOfClasses = 2  # Number of classes used for classification
trainingIterations = 50  # Number of epochs used for training
totalClients = 1  # Total number of clients expected for training
requestID = 22  # ID Number used to denote the request made on the WebPress page for training requests.
host = "127.0.0.1"  # IP address of the network interface that will listen for incoming connections. This parameter must be one of the IPv4 addresses that result from making an ifconfig call on the terminal (or ipconfig call on Windows)
max_udp_packet_size = 1500  # Maximum amount of bytes that the UDP socket/interface can accept. As a general rule, most interfaces and media accept MTU size of 1500 (maximum 1500 bytes per packet)
clients_number = 0  # Initialize the parameter for denoting client ID's
clients_list = []  # List of clients, to manipulate the fields and methods of each client seperately.
total_updates = 0  # Keeps the count of updates made on the model.
is_training = True  # Denotes whether the framework is training.
iteration_this = 0
accuracy_test = np.zeros([trainingIterations])  # List to hold the accuracy of the model at each epoch, used purely for benchmarking reasons.
port = 3001  # Clients are supposed to "connect" to this port.
central_model = {'weights': [], 'biases': []}
x_test = np.load('../worker/images_combined_worker.npy')  # Load up input data (test).
y_test = np.load('../worker/labels_combined_worker.npy')  # Load up label data (test).
y_test = tf.keras.utils.to_categorical(y_test, numberOfClasses)
x_test = np.reshape(x_test, (x_test.shape[0], -1)) / 255  # Reshape and normalize the test data
x_test = tf.convert_to_tensor(x_test, dtype=np.float32)  # Convert test data to tensors
y_test = tf.convert_to_tensor(y_test, dtype=np.float32)  # Convert test labels into tensors
curr_dir = os.curdir
global_model_file_name_prefix = "global_model"
global_model_file_handle = open(global_model_file_name_prefix + ".json", "w")
global_model_file_handle.truncate(0)
json_obj = {"weights": [], "biases": []}
json.dump(json_obj, global_model_file_handle)
global_model_file_handle.close()
mac_address = get_mac_address(ip=host)
local_model_file_prefix = "local_model_client_"
mk5_used = True
local_interface_address = "192.168.1.57"
remote_interface_address = "192.168.1.31"
local_port = 4001
remote_port = 4000
curr_date_time_format = "%d/%m/%Y, %H:%M:%S"
useful_characters = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "a", "b", "c", "d", "e", "f",  "g", "h", "i", "j", "k", "l", "m", "n", "o", "p",  "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "{", "}", "[", "]", ":", "-", ",", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", " ", "\"", "-", ".", ":"]


def regenerate_the_model(model) -> dict:
    try:
        model['weights'] = [tf.Variable(tf.convert_to_tensor(y)) for y in model['weights']]
    except ValueError:
        model['weights'] = []
        for i in range(numberOfLayers):
            model['weights'].append([])
            for j in range(numberOfNeurons[i]):
                model['weights'][i].append(generate_list_of_floats_of_certain_length(numberOfNeurons[i + 1]))
    except KeyError:
        model = {"weights": [], "biases": []}
        for i in range(numberOfLayers):
            model['weights'].append([])
            for j in range(numberOfNeurons[i]):
                model['weights'][i].append(generate_list_of_floats_of_certain_length(numberOfNeurons[i + 1]))

    try:
        model['biases'] = [tf.Variable(tf.convert_to_tensor(y)) for y in model['biases']]
    except ValueError:
        model['biases'] = []
        for i in range(numberOfLayers):
            model['biases'].append([])
            for j in range(numberOfNeurons[i]):
                model['biases'][i].append(
                    generate_list_of_floats_of_certain_length(numberOfNeurons[i + 1]))
    except KeyError:
        for i in range(numberOfLayers):
            model['biases'].append([])
            for j in range(numberOfNeurons[i]):
                model['biases'][i].append(
                    generate_list_of_floats_of_certain_length(numberOfNeurons[i + 1]))
    return model


def strip_away_indicators(input_string: str = "", indicator_string: str = "ha") -> str:
    first_indicator_found = input_string.find(indicator_string)
    print(first_indicator_found)
    input_string = input_string[first_indicator_found:]
    indicators_removed = False
    while not indicators_removed:
        if input_string[0] == "a":
            input_string = input_string.replace("a", "", 1)
        elif input_string[0] == "h":
            input_string = input_string.replace("h", "", 1)
        else:
            indicators_removed = True
    second_indicator_found = input_string.find("ha")
    if input_string[second_indicator_found + 1] != "a":
        second_indicator_found = input_string.find("ha", second_indicator_found + 1)
        input_string = input_string[0:second_indicator_found]
        pass
    else:
        input_string = input_string[0:second_indicator_found]
    return input_string


def ord_long_string(some_long_string: str = "") -> bytearray:
    returned_list = bytearray()
    for i in range(len(some_long_string)):
        returned_list.append(ord(some_long_string[i]) & 0xff)
    return returned_list


def chr_long_string(some_byte_array_arg: bytearray) -> str:
    temp_str = ""
    some_bytes_object = bytes(some_byte_array_arg)
    msg_byte_list = list(some_byte_array_arg)
    msg_char_list = []
    for i in range(len(msg_byte_list)):
        msg_char_list.append(chr(msg_byte_list[i]))
    return ''.join(msg_char_list)


def keep_useful_characters(str_to_parse: str = ""):
    str_to_parse_len = len(str_to_parse)
    index = 0
    while index < str_to_parse_len:
        if str_to_parse[index] in useful_characters:
            index += 1
            continue
        else:
            str_to_parse = str_to_parse.replace(str_to_parse[index], "")
            str_to_parse_len = len(str_to_parse)
    return str_to_parse


class CoordinatorSocket:
    def __init__(self, local_ip_address_arg: str = local_interface_address, remote_ip_address_arg: str = remote_interface_address, local_port_arg: int = local_port, remote_port_arg: int = remote_port):
        self.recvd_message = None
        self.local_ip_address = local_ip_address_arg
        self.remote_ip_address = remote_ip_address_arg
        self.local_port = local_port_arg
        self.remote_port = remote_port_arg
        self.udp_socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
        self.udp_socket.settimeout(5.0)
        self.udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        self.udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self.udp_socket.bind((self.local_ip_address, self.local_port))
        self.receiving = False
        self.sending = False
        self.idle = False
        self.model_to_send = dict()
        self.model_to_send_str = ""
        self.received_model = dict()
        self.received_model_str = ""
        self.packets_to_send = []
        self.received_packets = []
        self.max_string_length_per_packet = 500
        self.last_received_packet = ""
        self.received_packet_curr = ""

    def send_single_packet_to_router(self, curr_packet: str = "", pkt_len: int = 700) -> None:
        print("Test source to MK1 on %s:%d" % (self.remote_ip_address, self.remote_port))
        pkt_count = 0
        # Open the socket to communicate with the mk1
        Target = (self.remote_ip_address, self.remote_port)  # the target address and port
        self.udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        try:
            pktbuf = bytearray()
            some_string = "hahahahahahahahahahahahahahahaha" + curr_packet + "hahahaha"  # For some reason, the router eats away 27 characters from the string to send. Possibly due to the UDP headers and fields overall taking up 24 bytes and 3 bytes used for some other stuff.
            for i in range(len(some_string)):
                pktbuf.append(ord(some_string[i]) & 0xff)
            for i in range(0, pkt_len - len(some_string)):
                pktbuf.append(0 & 0xff)

            # print_byte_string(pktbuf)
            print('Total packets transmitted: %d' % (pkt_count + 1))

            # Transmit the packet to the MK1 via the socket
            self.udp_socket.sendto(pktbuf, Target)

            # Increment packet numbers
            pkt_count = pkt_count + 1
            print("Transmitted %d packets" % pkt_count)
            time.sleep(0.1)
            return
        except KeyboardInterrupt:
            self.udp_socket.close()
            print('User CTRL-C, stopping packet source')
            sys.exit(1)
        except:
            print("Got exception:", sys.exc_info()[0])
            raise

    def send_model(self) -> None:
        self.sending = True
        self.model_to_send_str = str(self.model_to_send)
        self.packets_to_send = textwrap.wrap(self.model_to_send_str, self.max_string_length_per_packet)
        index = 0
        adet = 0
        time.sleep(10.0)
        while index < len(self.packets_to_send):
            try:
                if mk5_used:
                    self.send_single_packet_to_router(curr_packet=self.packets_to_send[index])

                else:
                    self.udp_socket.sendto(self.packets_to_send[index].encode(),
                                           (self.remote_ip_address, self.remote_port))
                recvd_message, recvd_from = self.udp_socket.recvfrom(65536)
                if mk5_used:
                    if strip_away_indicators(chr_long_string(bytearray(recvd_message))) != "":
                        print("ACK received from ", recvd_from)
                        adet += 1
                        print("Sent: ", adet)
                    else:
                        continue
                else:
                    if recvd_message.decode() != "":
                        print("ACK received from ", recvd_from)
                    else:
                        continue
                index += 1
            except socket.timeout:
                print("Coordinator sending timed out...")
                continue
        for i in range(3):
            if mk5_used:
                self.send_single_packet_to_router(curr_packet="done")
            else:
                self.udp_socket.sendto("done".encode(), (self.remote_ip_address, self.remote_port))
            time.sleep(1.0)
        print("Model sent.")
        self.sending = False

    def receive_model(self) -> dict:
        self.received_packets = []
        self.received_model = dict()
        self.receiving = True
        received_packet_count = 0
        adet_re = 0
        repeat_packet_count = 0
        while self.receiving:
            try:
                self.recvd_message, recvd_from = self.udp_socket.recvfrom(65536)
                if mk5_used:
                    self.received_packet_curr = strip_away_indicators(chr_long_string(bytearray(self.recvd_message)))
                    print(self.received_packet_curr)
                else:
                    self.received_packet_curr = self.recvd_message.decode()
                    print(self.received_packet_curr)
                print("Packet received. Sending ACK...")
                if mk5_used:
                    self.send_single_packet_to_router(curr_packet="ACK", pkt_len=1600)
                else:
                    self.udp_socket.sendto("ACK".encode(), (self.remote_ip_address, self.remote_port))

                if "done" in self.last_received_packet:
                    break
                if self.last_received_packet == self.received_packet_curr:
                    print("Repeat packet obtained")
                    print("Received ", received_packet_count, "many unique packets, received ", repeat_packet_count,
                          " many repeat packets.")
                    repeat_packet_count += 1
                    continue
                self.last_received_packet = self.received_packet_curr
                self.received_packets.append(self.received_packet_curr)
                received_packet_count += 1
            except socket.timeout:
                continue
        print("Receiving finished.")
        print("Received ", received_packet_count, " many packets")
        print("Received ", repeat_packet_count, " many repeat packets.")
        index = 0
        received_packet_list_len = len(self.received_packets)
        while index < received_packet_list_len:
            if "done" in self.received_packets[index]:
                self.received_packets.remove(self.received_packets[index])
                received_packet_list_len = len(self.received_packets)
            elif chr(0) in self.received_packets[index]:
                self.received_packets.remove(self.received_packets[index])
                received_packet_list_len = len(self.received_packets)
            elif "ACK" in self.received_packets[index]:
                self.received_packets.remove(self.received_packets[index])
                received_packet_list_len = len(self.received_packets)
            else:
                index += 1
        print("Received packet list has length ", len(self.received_packets))
        received_file_handle = open("received_model.txt", "a+")
        received_file_handle.truncate(0)
        received_file_handle.write("".join(self.received_packets))
        received_file_handle.close()
        while "weights" not in self.received_packets[0]:
            self.received_packets.remove(self.received_packets[0])
        if len(self.received_packets) == 0:
            self.received_model = regenerate_the_model(self.received_model)
            self.receiving = False
            return self.received_model
        else:
            self.received_model_str = "".join(self.received_packets)
        try:
            self.received_model = eval(self.received_model_str)
        except:
            regenerate_the_model(self.received_model)
        self.receiving = False
        return self.received_model


def generate_list_of_floats_of_certain_length(length: int = 100) -> list:
    some_list = []
    for i in range(length):
        some_list.append(round(random.random(),))
    return some_list


class ClientThread(Thread):
    def __init__(self, this_id: int = clients_number, global_model_arg: dict = None, local_model_arg: dict = None, global_model_location_arg: str = (global_model_file_name_prefix + ".json")):
        Thread.__init__(self)
        self.id = this_id
        self.socket = CoordinatorSocket()
        self.global_model = Any
        if global_model_arg is None:
            self.global_model = central_model
        else:
            self.global_model = global_model_arg
        self.client_thread_local_model = dict()
        if local_model_arg is None:
            self.client_thread_local_model = {"weights": [], "biases": []}
        else:
            self.client_thread_local_model = local_model_arg
        self.model_str = str()
        self.local_model_str = str()
        self.waiting = True
        self.updated = True
        self.local_model_json_file_location = local_model_file_prefix + str(clients_number) + ".json"
        self.local_model_json_file = open(self.local_model_json_file_location, "a+")
        self.local_model_json_file.truncate(0)
        self.global_model_json_file_location = global_model_location_arg
        temp_file_handle = open(self.global_model_json_file_location, "a+")
        temp_file_handle.truncate(0)
        json.dump(self.global_model, temp_file_handle)
        temp_file_handle.close()
        self.global_model_json_file_handle = open(self.global_model_json_file_location, "r")
        self.global_model = json.load(self.global_model_json_file_handle)
        self.global_model_json_file_handle.close()
        self.iteration_number = 0


    def run(self):
        global total_updates, iteration_this
        while True:
            if self.iteration_number < trainingIterations:
                if self.waiting & self.updated:  # client is downloading the model
                    self.socket.model_to_send = central_model
                    while self.socket.receiving or self.socket.sending:
                        time.sleep(1.0)
                    self.socket.send_model()
                    self.local_model_str = str(self.client_thread_local_model)
                    self.updated = False
                    self.waiting = False
                elif not self.updated:  # client is uploading the model
                    # self.client_thread_local_model = self.receiving_socket.receive_udp_packet()
                    while self.socket.receiving or self.socket.sending:
                        time.sleep(1.0)
                    self.client_thread_local_model = self.socket.receive_model()
                    total_updates += 1
                    self.iteration_number += 1
                    self.updated = True
            else:
                break
        print("Client ", self.id, " is disconnected.")
        self.socket.udp_socket.close()


def model_average(client_list, model_global):
    global numberOfLayers
    sum_clients_biases = list(list())  # Create jagged array for client's biases.
    for layer_ in range(numberOfLayers):
        sum_clients_weights = np.zeros(shape=(numberOfNeurons[layer_], numberOfNeurons[layer_ + 1]))
        print(type(sum_clients_weights))
        for i in range(numberOfNeurons[layer_]):
            for curr_client in client_list:
                try:
                    sum_clients_weights[i] += curr_client.client_thread_local_model['weights'][layer_][i]
                except:
                    curr_client.client_thread_local_model = regenerate_the_model(model=curr_client.client_thread_local_model)
                    sum_clients_weights[i] += curr_client.client_thread_local_model['weights'][layer_][i]
        model_global['weights'][layer_] = np.divide(sum_clients_weights, totalClients).tolist()
    for i in range(1, len(numberOfNeurons)):
        sum_clients_biases.append(np.zeros(shape=(numberOfNeurons[i])).tolist())
    for i in range(len(sum_clients_biases)):
        for k in range(len(sum_clients_biases[i])):
            for current_client in client_list:
                try:
                    sum_clients_biases[i][k] += current_client.client_thread_local_model['biases'][i][k] / totalClients
                except:
                    current_client.client_thread_local_model = regenerate_the_model(model=current_client.client_thread_local_model)
                    sum_clients_biases[i][k] += current_client.client_thread_local_model['biases'][i][k] / totalClients
    model_global['biases'] = sum_clients_biases
    return model_global


def neural_net(x, model):
    y = [x]
    for layer_ in range(numberOfLayers):
        if activationFunction[layer_] == 'relu':
            try:
                y.append(tf.nn.relu(tf.matmul(y[layer_], model['weights'][layer_]) + model['biases'][layer_]))
            except:
                regenerate_the_model(model=model)
                y.append(tf.nn.relu(tf.matmul(y[layer_], model['weights'][layer_]) + model['biases'][layer_]))
        elif activationFunction[layer_] == 'softmax':
            try:
                y.append(tf.nn.softmax(tf.matmul(y[layer_], model['weights'][layer_]) + model['biases'][layer_]))
            except:
                regenerate_the_model(model=model)
                y.append(tf.nn.softmax(tf.matmul(y[layer_], model['weights'][layer_]) + model['biases'][layer_]))
    return y[-1]


def accuracy(y_predicted, y_true):
    # Predicted class is the index of highest score in prediction vector (i.e. argmax).
    correct_prediction = tf.equal(tf.argmax(y_predicted, 1), tf.argmax(y_true, 1))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis=-1)


for layer in range(numberOfLayers):  # Initialize the global model. Not going to be used multiple times.
    weight_layer_to_add = tf.Variable(tf.random.truncated_normal([numberOfNeurons[layer], numberOfNeurons[layer + 1]], stddev=0.1)).read_value().numpy().tolist()
    central_model["weights"].append(weight_layer_to_add)
    bias_layer_to_add = tf.Variable(tf.zeros([numberOfNeurons[layer + 1]])).read_value().numpy().tolist()
    central_model["biases"].append(bias_layer_to_add)

global_model_file_handle = open((global_model_file_name_prefix + ".json"), "a+")
global_model_file_handle.truncate(0)
json.dump(central_model, global_model_file_handle)
global_model_file_handle.close()

while clients_number < totalClients:
    client = ClientThread(clients_number)
    clients_list.append(client)
    clients_number += 1


print("Training started...")


for client_this in clients_list:
    client_this.start()

iterations = []
while is_training:
    try:
        if not total_updates < totalClients:  # all models are updated
            central_model = model_average(clients_list, central_model)
            for client_this in clients_list:
                client_this.global_model = central_model
                client_this.waiting = True
            iteration_this += 1
            iterations.append(iteration_this)
            is_training = iteration_this < trainingIterations
            print(iteration_this)
            total_updates = 0
            test_predicted = neural_net(x_test, central_model)
            test_Accuracy = accuracy(test_predicted, y_test)
            accuracy_test[iteration_this - 1] = test_Accuracy
    except KeyboardInterrupt:
        sys.exit(0)

testAccuracy = {'xaxis': iterations, 'yaxis': accuracy_test.tolist()}
results = {'name': 'testAccuracy', 'data': testAccuracy}
results_dict = {'results': results}

with open('results_' + str(requestID) + '.json', 'w') as fp:
    json.dump(results_dict, fp)

while total_updates < totalClients:  # inefficient ending
    time.sleep(.1)

exit_iteration = 0
for client_this in clients_list:
    exit_iteration += 1
    client_this.updated = False

print("Training finished.")
