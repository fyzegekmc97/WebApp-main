import json
import os
import socket
import sys
import textwrap
import numpy as np
import random
import time
import numpy.random as npyrnd
from numpy import dtype as npydtype
from coordinator_udp_sender import Smart_UDP_Sender_Socket
from keras import models, layers
from typing import List
# from coordinator_udp_receiver import AlwaysOpenUDP_Receiver
# import paramiko
import select
import zipfile
import tensorflow as tf
import csv
import pathlib
import cv2
import pickle
import communication
import srudp


useful_characters = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "a", "b", "c", "d", "e", "f",  "g", "h", "i", "j", "k", "l", "m", "n", "o", "p",  "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "{", "}", "[", "]", ":", "-", ",", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", " ", "\"", "-", ".", ":"]


rand_string = ""
test_string = "test_string"
rand_text_length = 4500
string_length_per_packet = 1500


for i in range(rand_text_length):
    rand_string += "1"
rand_packets = textwrap.wrap(rand_string, string_length_per_packet)


def read_json_file():
    some_file_handle = open("results_22.json", "r")
    my_json_obj = json.load(some_file_handle)
    print(my_json_obj)


def read_already_open_file():
    some_file = open("test_file.txt", "r")
    some_file_copy = open("test_file.txt", "r")
    print(some_file)
    print(some_file_copy)
    print("Done")


def change_directory_test():
    curr_dir = os.getcwd()
    print(curr_dir)
    os.chdir("../images/")
    curr_dir = os.getcwd()
    print(curr_dir)
    print(os.curdir)


def create_numpy_array():
    some_numpy_ndarray = np.zeros(shape=(1, 10), dtype=float)
    some_numpy_ndarray_converted_to_list = some_numpy_ndarray.tolist()
    print(some_numpy_ndarray_converted_to_list)


def add_two_numpy_arrays():
    first_numpy_array = np.zeros(shape=(1, 10), dtype=float)
    second_numpy_array = np.arange(10)
    print(first_numpy_array + second_numpy_array)


def get_multidimensional_array_length():
    some_dumb_array = list(list([]))
    some_dumb_array.append([1, 2, 3, 4])
    some_dumb_array.append([3, 4, 5, 6, 7])
    print(some_dumb_array[0])
    print(len(some_dumb_array[0]))


def generate_random_numbers():
    nanoseconds_time = time.perf_counter_ns()
    random.seed(nanoseconds_time)
    print(random.random(), random.random(), random.random())
    random.seed(nanoseconds_time)
    print(random.random(), random.random(), random.random())
    print(random.randint(15, 20))
    print(random.sample([1, 2, 3, 4, 5, 6, 7, 8], 3))
    a = [1, 2, 3, 4, 5, 6, 7]
    random.shuffle(a)
    print(a)
    print("Betavariate: ", random.betavariate(2, 2))
    print("Gauss: ", random.gauss(0, 1))
    npyrnd.seed(15)
    print(npyrnd.rand(3))
    npyrnd.seed(15)
    print(npyrnd.rand(3))
    print(npyrnd.randint(low=55, high=100, dtype=int))


def send_custom_payload():
    remote_port = 5757
    remote_ip = "192.168.1.41"
    remote_address_tuple = (remote_ip, remote_port)
    custom_payload = "Some payload"
    packet_data = custom_payload.encode()
    some_socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
    expected_byte_length = 1500
    packet = bytearray()
    packet.extend(bytearray(packet_data))
    while True:
        try:
            for i in range(expected_byte_length - len(packet_data) - 8):
                packet.extend(str(0).encode())
            some_socket.sendto(packet, remote_address_tuple)
        except socket.timeout:
            continue
        except KeyboardInterrupt:
            break


def send_reliable_data_router(remote_addr: str = "192.168.1.41", remote_port: int = 4040, pkt_rate=50, max_bytes_per_packet=1500, local_port_to_send_from: int = 5757, socket_arg: socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM) = None):

    PktPeriod = 1.0 / pkt_rate  # Period of packet generation
    sleeptime = PktPeriod  # time to sleep between packets

    print("Test source to MK1 on %s:%d" % (remote_addr, remote_port))
    rxsock = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
    if socket_arg is None:
        rxsock = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
    else:
        rxsock = socket_arg
    rxsock.bind(('', local_port_to_send_from))
    rxsock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    Target = (remote_addr, remote_port)  # the target address and port
    new_time = time.time()
    start_pkt_num = 0
    last_pkt_num = 0
    pkt_count = 0
    last_update_time = time.time()
    start_time = last_update_time
    UpdatePeriod = 1.0
    PktRatePeriod = 1.0
    all_packets_sent_reliably = False
    try:
        # Main loop
        while not all_packets_sent_reliably:
            pkt_generation_time = time.time()
            pktbuf = bytearray()
            some_string = "abc"
            for i in range(len(some_string)):
                pktbuf.append(ord(some_string[i]) & 0xff)
                print(chr(ord(some_string[i])))
            for i in range(0, max_bytes_per_packet - len(some_string)):
                pktbuf.append(0 & 0xff)

            # print_byte_string(pktbuf)
            print('Total packets transmitted: %d' % (pkt_count + 1))
            # print ('')

            # Transmit the packet to the MK1 via the socket
            rxsock.sendto(pktbuf, Target)

            # Increment packet numbers
            pkt_count = pkt_count + 1

            if new_time - last_update_time > UpdatePeriod:
                tx_pkt_rate = ((pkt_count - last_pkt_num) /
                               (new_time - last_update_time))
                last_pkt_num = pkt_count
                last_update_time = new_time

            last_time = new_time
            new_time = time.time()

            timediff = new_time - last_time

            errtime = (PktPeriod * (pkt_count - start_pkt_num) -
                       (new_time - start_time))
            sleeptime = PktPeriod + errtime

            if new_time - start_time > PktRatePeriod:
                start_time = new_time
                start_pkt_num = pkt_count

            if sleeptime < 0:
                sleeptime = 0
            time.sleep(sleeptime)

        print("Transmitted %d packets" % pkt_count)
        return pkt_count
    except KeyboardInterrupt:
        rxsock.close()
        print('User CTRL-C, stopping packet source')
        sys.exit(1)

    except:
        rxsock.close()
        print("Got exception:", sys.exc_info()[0])
        raise


def send_reliable_data_loopback():
    reliable_packet_index = 0
    sender = Smart_UDP_Sender_Socket(udp_local_address_arg=("127.0.0.1", 5000), udp_target_address_arg=("127.0.0.1", 5250))
    sender.bind_to_local_address()
    sender.initiate_handshake()


def ord_long_string(some_long_string: str = "") -> bytearray:
    returned_list = bytearray()
    for i in range(len(some_long_string)):
        returned_list.append(ord(some_long_string[i]) & 0xff)
    return returned_list


def chr_long_string(some_byte_array_arg: bytearray) -> str:
    temp_str = ""
    some_bytes_object = bytes(some_byte_array_arg)
    some_bytes_object_decoded = some_bytes_object.decode()
    print()
    return temp_str


def ack_router_test():
    sender = Smart_UDP_Sender_Socket(udp_local_address_arg=("192.168.1.57", 5000),
                                     udp_target_address_arg=("192.168.1.31", 5757))
    sender.bind_to_local_address()
    sender.send_ACK_to_router(0)


def send_test_string():
    sender = Smart_UDP_Sender_Socket(udp_local_address_arg=("192.168.1.57", 4002),
                                     udp_target_address_arg=("192.168.1.31", 4000))
    sender.bind_to_local_address()
    sender.send_single_packet_to_router(curr_packet=test_string)


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


def generate_list_of_floats_of_certain_length(length: int = 100) -> list:
    some_list = []
    for i in range(length):
        some_list.append(random.random())
    return some_list


def receive_test_string():
    some_socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
    some_socket.bind(("192.168.1.57", 4002))
    recvd_bytes, recvd_from = some_socket.recvfrom(1500)
    print(recvd_bytes)
    print(recvd_from)
    some_socket.close()
    some_socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
    some_socket.bind(("192.168.1.57", 4002))
    recvd_bytes, recvd_from = some_socket.recvfrom(1500)
    print(recvd_bytes)
    print(recvd_from)
    some_socket.close()


def remove_indicator_string(main_string: str = "mystr_some_mystr", indicator_string: str = "mystr"):
    main_string = main_string.replace(indicator_string, "")
    return main_string


def strip_away_indicators(input_string: str = "", indicator_string: str = "^||") -> str:
    indicator_found = input_string.find(indicator_string)
    input_string = input_string[indicator_found:]
    indicator_found = input_string.rfind(indicator_string)
    input_string = input_string[:indicator_found]
    indicator_found = input_string.find(indicator_string)
    indicator_found_reverse = input_string.rfind(indicator_string)
    while (indicator_found != indicator_found_reverse) and (indicator_found >= 0) and (indicator_found_reverse >= 0):
        input_string = input_string[indicator_found+len(indicator_string):]
        input_string = input_string[:indicator_found_reverse]
        indicator_found = input_string.find(indicator_string)
        indicator_found_reverse = input_string.rfind(indicator_string)
    input_string = input_string.replace(indicator_string, "")
    return input_string


def receive_long_string():
    test_socket = socket.socket()
    test_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    test_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    test_socket.bind(("127.0.0.1", 5000))
    test_socket.listen()
    received_packets = []
    print(remove_indicator_string())
    while True:
        try:
            test_socket_connection, test_received_from = test_socket.accept()
            print("Connection complete")
            break
        except:
            print("Exception raised...")
            sys.exit(0)
    while True:
        try:
            some_string = test_socket_connection.recv(2500)
            if type(some_string) is not bytes:
                print(type(some_string))
            if some_string is not None:
                if "mystr" in some_string.decode():
                    print("Received a string of length: ", len(some_string.decode().replace("mystr", "")))
                    print(some_string.decode().replace("mystr", ""))
                    received_packets.append(some_string.decode().replace("mystr", ""))
                    print("Received packet count: ", len(received_packets))
                    print("Received packets cumulative length: ", len("".join(received_packets)))

        except KeyboardInterrupt:
            sys.exit(0)
        except TypeError:
            continue
        except:
            print(sys.exc_info())
            time.sleep(1.0)
            continue


def neural_net():
    x_train_new = np.load("images50_percent_training_data.npy")
    y_train = np.load("labels50_percent_training_data.npy")
    x_test_new = np.load("images50_percent_testing_data.npy")
    y_test = np.load("labels50_percent_testing_data.npy")
    model = models.Sequential()
    model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=(200, 200, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(2))
    model.summary()
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    history = model.fit(x_train_new, y_train, epochs=1, validation_data=(x_test_new, y_test))
    model_json = model.to_json()
    print(model_json)
    print(type(model_json))
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = models.model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")
    print(loaded_model)
    print(type(loaded_model))
    print(loaded_model.weights)
    print(loaded_model.layers)
    list_of_files_to_zip = ["model.h5", "model.json"]
    with zipfile.ZipFile('model_files.zip', 'w') as zipMe:        
        for file in list_of_files_to_zip:
            zipMe.write(file, compress_type=zipfile.ZIP_DEFLATED)
    some_file = open("model_files.zip", "rb")
    my_bytes = some_file.read()
    print(len(my_bytes))
    sending_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sending_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sending_socket.bind(("127.0.0.1", 5000))
    sending_socket.connect(("127.0.0.1",5001))
    print("Connected")
    sending_socket.sendall(my_bytes)


def change_float_type():
    some_np_arr = np.random.random((15, 1))
    some_np_arr_f16 = some_np_arr.astype(dtype=np.float16)
    print(some_np_arr)
    print(some_np_arr_f16)
    some_np_arr_f16_f32 = some_np_arr_f16.astype(dtype=np.float32)
    print(some_np_arr_f16_f32)


def edge_detection():
    print(pathlib.Path.home())
    some_image = cv2.imread("photos_common_directory/no20220719-124848-000001fframe0.jpg")
    print(some_image.shape)
    some_image = cv2.resize(some_image, (200,200))
    print(some_image.shape)
    some_input = np.ndarray(shape=(1,200,200,3), dtype=np.float32)
    some_input[0] = some_image
    some_tensor = tf.convert_to_tensor(some_image)
    # output = tf.keras.layers.Conv2D(3, 3, activation='relu', input_shape=some_input.shape)(some_input)
    # print(output.shape)
    img_gray = cv2.cvtColor(some_image, cv2.COLOR_BGR2GRAY)
    print(type(img_gray[0][0]))
    img_blur = cv2.GaussianBlur(img_gray, (3,3), 0)
    edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)
    cv2.imshow('Canny Edge Detection', edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(edges.shape)
    print(type(edges[0][0]))
    print(edges)


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
    

def view_images_and_labels(images: np.ndarray, labels: np.ndarray):
    for image, label in zip(images,labels):
        cv2.imshow("Current image", image)
        print("Label is: ", label)
        pressed_key = cv2.waitKey(0)
        if pressed_key == ord("q"):
            cv2.destroyAllWindows()
            break
        cv2.destroyAllWindows()


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
            else:
                print("Did not get file size but got: ", decoded_bytes)
                continue
        except:
            pass


def receive_file(some_socket:socket.socket, file_length: int, packet_size: int = 1500):
    bytes_buffer = bytearray()
    while len(bytes_buffer) < file_length:
        recvd_bytes = some_socket.recv(packet_size)
        bytes_buffer.extend(recvd_bytes)
        print("Received bytes of length: ", len(recvd_bytes))
    return bytes_buffer


def receive_ID(some_socket: socket.socket, ID_keyword: str):
    recvd_bytes = some_socket.recv(1500)
    return int(recvd_bytes.decode().replace(ID_keyword, ""))


if __name__ == "__main__":
    my_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    my_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    my_socket.bind(("127.0.0.1", 2500))
    print("Socket bound")
    file_length = receive_file_size(my_socket, "file_len_")
    my_file_bytes = receive_file(my_socket, file_length)
    pass
