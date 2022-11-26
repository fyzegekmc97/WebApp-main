import random
import socket
import pickle
import sys
import os
import textwrap
import communication
import numpy as np
try:
    import tensorflow as tf
except KeyboardInterrupt:
    sys.exit(0)
import time
import json
from getmac import get_mac_address
from threading import Thread
from worker_udp_receiver import Smart_UDP_Receiver
from worker_udp_sender import Smart_UDP_Sender_Socket
import signal


numberOfLayers = 3
activationFunction = ['relu', 'relu', 'softmax']
numberOfEpochs = 1
learningRate = 0.001
numberOfClasses = 2
host = "127.0.0.1"  # Downlink socket address (remote)
testing_proxy_code = False
got_dict_at_init = False
global_model_file_prefix = "global_model"
local_model_file_prefix = "local_model"
curr_date_time_format = "%d/%m/%Y, %H:%M:%S"
mac_address = get_mac_address(ip=host)
totalClients = 1  # Total number of clients expected for training
num_of_concurrent_clients_on_device = 1
print("Number of concurrent clients on device: ", num_of_concurrent_clients_on_device)
local_port = 4003
remote_port = 4000
client_local_id = 0
workers_list = []
numberOfNeurons = [784, 128, 64, 2]  # length of this array must be numberOfLayers + 1
mk5_used = True
local_ip_address = "192.168.1.57"
remote_ip_address = "192.168.1.41"
x_train = np.load('images_combined_worker.npy')
y_train = np.load('labels_combined_worker.npy')
numberOfMiniBatches = x_train.shape[0]
miniBatchSize = int(numberOfMiniBatches / 10)
y_train = tf.keras.utils.to_categorical(y_train, numberOfClasses)
x_train = np.reshape(x_train, (x_train.shape[0], -1)) / 255
iteration_count = 50
useful_characters = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "a", "b", "c", "d", "e", "f",  "g", "h", "i", "j", "k", "l", "m", "n", "o", "p",  "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "{", "}", "[", "]", ":", "-", ",", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", " ", "\"", "-", ".", ":"]
adet = 0


def keep_useful_characters(str_to_parse: str = "") -> str:
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


class UDP_Sender_Thread(Thread):
    def __init__(self):
        Thread.__init__(self)
        self.sender = Smart_UDP_Sender_Socket()

    def run(self) -> None:
        try:
            self.sender.start_sending()
        except KeyboardInterrupt:
            self.sender.should_run = False
            self.sender.execution_stopping_reason = "Keyboard interrupt"
            os.kill(os.getpid(), signal.SIGKILL)
            time.sleep(1)


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


class WorkerSocket:
    def __init__(self, local_ip_address_arg: str = local_ip_address, remote_ip_address_arg: str = remote_ip_address, local_port_arg: int = local_port, remote_port_arg: int = remote_port):
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
                    adet += 1
                    print("sended", adet)
                else:
                    self.udp_socket.sendto(self.packets_to_send[index].encode(),
                                           (self.remote_ip_address, self.remote_port))
                recvd_message, recvd_from = self.udp_socket.recvfrom(65536)
                if mk5_used:
                    if strip_away_indicators(chr_long_string(bytearray(recvd_message))) != "":
                        print("ACK received from ", recvd_from)
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
        self.sending = False
        print("Model sent.")

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
                    print("Received ", received_packet_count, "many unique packets, received ", repeat_packet_count, " many repeat packets.")
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
        try:
            while "weights" not in self.received_packets[0]:
                self.received_packets.remove(self.received_packets[0])
        except IndexError:
            self.receive_model()
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
        some_list.append(random.random())
    return some_list


class Worker(Thread):
    def __init__(self, client_local_id_arg: int = 0, client_global_id: int = 0, bypass_global_id_reception: bool = False):
        Thread.__init__(self)
        self.global_id = client_global_id
        self.local_id = client_local_id_arg
        self.socket = WorkerSocket()
        self.iteration_number = 0

    def run(self) -> None:
        global got_dict_at_init
        try:
            while True:
                # download the global model first
                print("Running iteration number: ", self.iteration_number)
                while self.socket.receiving or self.socket.sending:
                    time.sleep(1.0)
                message_decoded = self.socket.receive_model()
                # Proxy code is disabled for now...
                if testing_proxy_code:  # Not going to activate
                    if got_dict_at_init:
                        message_decoded = pickle.loads(message)
                    else:
                        while True:
                            got_dict_at_init = False
                            message = communication.receive_data(clientSocket)
                            if len(message) == 0 or message is None:
                                continue
                            else:
                                try:
                                    message_decoded = pickle.loads(message)
                                    print("Received message of type: ", type(message_decoded))
                                    print("Received model of size", len(message_decoded))
                                except TypeError:
                                    print("Type error happened")
                                    (exception_type, exception_value, exception_traceback) = sys.exc_info()
                                    print("Exception information: ")
                                    print("Exception type: ", exception_type)
                                    print("Exception value: ", exception_value)
                                    print("Exception traceback:", exception_traceback)
                                break
                    got_dict_at_init = False

                if self.iteration_number < iteration_count:
                    print(self.iteration_number)

                    # model initialization
                    model = message_decoded  # Actually the global model downloaded from the parameter server

                    for epoch in range(numberOfEpochs):
                        batches_x, batches_y = mini_batches(x_train, y_train, miniBatchSize,
                                                            numberOfMiniBatches)  # Obtain the batches to be used for this training epoch.
                        for batch_x, batch_y in zip(batches_x, batches_y):
                            # convert the weight to variable
                            try:
                                model['weights'] = [tf.Variable(tf.convert_to_tensor(y)) for y in model['weights']]  # Updating the local model according to the received global model.
                            except ValueError:
                                for i in range(len(model['weights'])):
                                    for k in range(len(model['weights'][i])):
                                        if len(model['weights'][i][k]) != numberOfNeurons[i+1]:
                                            model['weights'][i][k] = np.random.uniform(low=0.0, high=1.0, size=(numberOfNeurons[i+1],)).tolist()
                                try:
                                    model['weights'] = [tf.Variable(tf.convert_to_tensor(y)) for y in model['weights']]
                                except ValueError:
                                    model['weights'] = []
                                    for i in range(numberOfLayers):
                                        model['weights'].append([])
                                        for j in range(numberOfNeurons[i]):
                                            model['weights'][i].append(generate_list_of_floats_of_certain_length(numberOfNeurons[i+1]))
                            except KeyError:
                                model = regenerate_the_model(model)
                                model['weights'] = [tf.Variable(tf.convert_to_tensor(y)) for y in model['weights']]  # Updating the local model according to the received global model.

                            try:
                                model['biases'] = [tf.Variable(tf.convert_to_tensor(y)) for y in model['biases']]  # Updating the local model according to the received global model.
                            except ValueError:
                                for i in range(len(model['biases'])):
                                    for k in range(len(model['biases'][i])):
                                        if len(model['biases'][i][k]) != numberOfNeurons[i+1]:
                                            model['biases'][i][k] = np.random.uniform(low=0.0, high=1.0, size=(numberOfNeurons[i+1],)).tolist()
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
                                model = regenerate_the_model(model)
                                model['biases'] = [tf.Variable(tf.convert_to_tensor(y)) for y in model['biases']]  # Updating the local model according to the received global model.
                            # compute gradient
                            try:
                                train_predicted = neural_net(batch_x, model)  # Updating the local model according to the received global model.
                            except:
                                pass
                            gradients, loss = get_gradients(batch_x, batch_y, model)  # Updating the local model according to the received global model.
                            # local update of the model
                            for layer_ in range(numberOfLayers):
                                model['weights'][layer_] = model['weights'][layer_] - learningRate * gradients['weights'][layer_]  # Updating the local model according to the received global model.
                                model['biases'][layer_] = model['biases'][layer_] - learningRate * gradients['biases'][layer_]  # Updating the local model according to the received global model.
                    print("Model is of type: ", type(model))
                    print("The model in the worker takes up memory size of: ", sys.getsizeof(model))
                    model_str = str(model)
                    model_str_len = len(model_str)
                    temp_file_handle = open((local_model_file_prefix + "_calculated_at_iteration_" + str(self.iteration_number) + ".json"), "a+")
                    temp_file_handle.truncate(0)
                    temp_model = {"weights": [], "biases": []}
                    for i in range(len(model['weights'])):
                        temp_model['weights'].append(model['weights'][i].numpy().tolist())
                    for i in range(len(model['biases'])):
                        temp_model['biases'].append(model['biases'][i].numpy().tolist())
                    json.dump(temp_model, temp_file_handle)
                    temp_file_handle.close()
                    temp_file_handle = open((local_model_file_prefix + "_to_send" + str(self.local_id) + ".json"), "a+")
                    temp_file_handle.truncate(0)
                    json.dump(temp_model, temp_file_handle)
                    temp_file_handle.close()
                    self.socket.model_to_send = temp_model
                    while self.socket.sending or self.socket.receiving:
                        time.sleep()
                    self.socket.send_model()
                    self.iteration_number += 1
                else:
                    break
        except KeyboardInterrupt:
            print("Keyboard interrupt raised.")
            sys.exit(0)


def neural_net(x, model_):
    y = [x]
    for layer_ in range(numberOfLayers):
        if activationFunction[layer_] == 'relu':
            try:
                y.append(tf.nn.relu(tf.matmul(y[layer_], model_['weights'][layer_]) + model_['biases'][layer_]))
            except:
                print(sys.exc_info())
                pass
        elif activationFunction[layer_] == 'softmax':
            try:
                y.append(tf.nn.softmax(tf.matmul(y[layer_], model_['weights'][layer_]) + model_['biases'][layer_]))
            except:
                print(sys.exc_info())
                pass
    return y[-1]


def mini_batches(X, Y, batch_size, numberOfMiniBatches=int(3)):
    m = X.shape[0]
    numberOfMiniBatches = min(numberOfMiniBatches, m // batch_size)
    rng = np.random.default_rng()
    perm = list(rng.permutation(m))
    x_batches, y_batches = list(), list()
    x_temp = X[perm, :]
    y_temp = Y[perm, :].reshape((m, Y.shape[1]))
    for i in range(numberOfMiniBatches):
        x_batches.append(
            tf.convert_to_tensor(x_temp[i * batch_size:(i + 1) * batch_size, :], dtype=np.float32))
        y_batches.append(
            tf.convert_to_tensor(y_temp[i * batch_size:(i + 1) * batch_size, :], dtype=np.float32))
    return x_batches, y_batches


def cross_entropy(y_predicted, y_true):
    # Clip prediction values to avoid log(0) error.
    y_predicted = tf.clip_by_value(y_predicted, 1e-9, 1.)
    # Compute cross-entropy.
    return -tf.reduce_sum(y_true * tf.math.log(y_predicted))


def accuracy(y_predicted, y_true):
    # Predicted class is the index of highest score in prediction vector (i.e. argmax).
    correct_prediction = tf.equal(tf.argmax(y_predicted, 1), tf.argmax(y_true, 1))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis=-1)


def list_to_dict(model_dict, model_list):
    model_dict_new = model_dict.copy()
    item_counter = 0
    for key in model_dict_new.keys():
        number_items = len(model_dict_new[key])
        this_list = [model_list[index] for index in range(item_counter, item_counter + number_items)]
        model_dict_new[key] = this_list
        item_counter += number_items

    return model_dict_new


def regenerate_the_model(model) -> dict:
    model = {"weights": [], "biases": []}
    try:
        model['weights'] = [tf.Variable(tf.convert_to_tensor(y)) for y in model['weights']]
    except:
        model['weights'] = []
        for i in range(numberOfLayers):
            model['weights'].append([])
            for j in range(numberOfNeurons[i]):
                model['weights'][i].append(generate_list_of_floats_of_certain_length(numberOfNeurons[i + 1]))

    try:
        model['biases'] = [tf.Variable(tf.convert_to_tensor(y)) for y in model['biases']]
    except:
        model['biases'] = []
        for i in range(numberOfLayers):
            model['biases'].append([])
            for j in range(numberOfNeurons[i]):
                model['biases'][i].append(
                    generate_list_of_floats_of_certain_length(numberOfNeurons[i + 1]))

    return model


def get_gradients(x, y, model_):
    # Variables to update, i.e. trainable variables.
    trainable_variables = model_['weights'] + model_['biases']

    with tf.GradientTape() as g:
        y_predicted = [x]
        for layer_ in range(numberOfLayers):
            if activationFunction[layer_] == 'relu':
                try:
                    y_predicted.append(
                        tf.nn.relu(
                            tf.matmul(y_predicted[layer_], model_['weights'][layer_]) + model_['biases'][layer_]))
                except:
                    model_ = regenerate_the_model(model=model_)
                    y_predicted.append(tf.nn.relu(tf.matmul(y_predicted[layer_], model_['weights'][layer_]) + model_['biases'][layer_]))
            elif activationFunction[layer_] == 'softmax':
                try:
                    y_predicted.append(
                        tf.nn.softmax(
                            tf.matmul(y_predicted[layer_], model_['weights'][layer_]) + model_['biases'][layer_]))
                except:
                    model_ = regenerate_the_model(model=model_)
                    y_predicted.append(
                        tf.nn.relu(
                            tf.matmul(y_predicted[layer_], model_['weights'][layer_]) + model_['biases'][layer_]))

        predictions = y_predicted[-1]
        loss_this = cross_entropy(predictions, y)
        if loss_this > 5:
            loss_this = tf.random.uniform(shape=(1,), dtype=tf.float32)
        # Compute gradients.
        for i in range(len(trainable_variables)):
            trainable_variables[i] = tf.Variable(trainable_variables[i], trainable=True)
        loss_this = tf.convert_to_tensor(loss_this, dtype=tf.float32)
        gradients_ = g.gradient(loss_this, trainable_variables)
        for i in range(numberOfLayers):
            if gradients_[i] is None:
                gradients_[i] = tf.random.uniform(shape=(numberOfNeurons[i], numberOfNeurons[i+1]), minval=-1, maxval=1, dtype=tf.float32)
        for i in range(numberOfLayers):
            if gradients_[i+numberOfLayers] is None:
                gradients_[i+numberOfLayers] = tf.random.uniform(shape=(numberOfNeurons[i+1],), minval=-1, maxval=1, dtype=tf.float32)
        gradients_dict = list_to_dict(model_, gradients_)
    return gradients_dict, loss_this


clientSocket = socket.socket()

while client_local_id < num_of_concurrent_clients_on_device:
    worker = Worker(client_local_id_arg=client_local_id)
    workers_list.append(worker)
    client_local_id += 1

for current_worker in workers_list:
    current_worker.start()

for current_worker in workers_list:
    current_worker.join()



