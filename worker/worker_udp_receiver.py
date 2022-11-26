import json
import os
import socket
import threading
import time
from threading import *
from typing import *
from datetime import datetime
import logging
import sys
import pickle
import signal
try:
    import tensorflow as tf
except KeyboardInterrupt:
    sys.exit(0)
import numpy as np
import multiprocessing


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


curr_date_time_format = "%d/%m/%Y, %H:%M:%S"
useful_characters = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "a", "b", "c", "d", "e", "f",  "g", "h", "i", "j", "k", "l", "m", "n", "o", "p",  "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "{", "}", "[", "]", ":", "-", ",", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", " ", "\"", "-", ".", ":"]
numberOfLayers = 3  # Total number of layers expected within the neural network
activationFunction = ['relu', 'relu', 'softmax']  # Activation functions of each layer
numberOfNeurons = [784, 128, 64, 2]  # length of this array must be numberOfLayers + 1


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


class Smart_UDP_Receiver_Thread(Thread):
    def __init__(self):
        Thread.__init__(self)
        self.receiver = Smart_UDP_Receiver()

    def run(self) -> None:
        self.receiver.receive_udp_packet()
        print("One receival finished.")


class AlwaysOpenUDP_Receiver(Thread):
    def __init__(self, local_ip_address: str = "192.168.1.57", remote_ip_address: str = "192.168.1.41", local_port: int = 4001, remote_port: int = 4000):
        Thread.__init__(self)
        self.udp_socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
        self.local_ip = local_ip_address
        self.local_port = local_port
        self.remote_ip = remote_ip_address
        self.remote_port = remote_port
        self.udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        self.udp_socket.bind((self.local_ip, self.local_port))
        self.received_packets = []
        self.udp_socket.settimeout(1.0)
        self.receiving_from_mk5 = True
        self.should_exit = False
        self.idle = True
        self.receiving = False
        self.last_received_string = ""
        self.ready_to_read = False
        self.last_received_packet = ""

    def send_single_packet_to_router(self, curr_packet: str = "", pkt_len: int = 1500):
        print("Test source to MK1 on %s:%d" % (self.remote_ip, self.remote_port))
        Target = (self.remote_ip, self.remote_port)
        pkt_count = 0
        try:
            pktbuf = bytearray()
            some_string = "abcdefghijklmnopqrstuvwxyzahahahaha" + curr_packet + "hahahah"  # For some reason, the router eats away 27 characters from the string to send. Possibly due to the UDP headers and fields overall taking up 24 bytes and 3 bytes used for some other stuff.
            for i in range(len(some_string)):
                pktbuf.append(ord(some_string[i]) & 0xff)
            for i in range(0, pkt_len - len(some_string)):
                pktbuf.append(0 & 0xff)
            print('Total packets transmitted: %d' % (pkt_count + 1))
            self.udp_socket.sendto(pktbuf, Target)
            pkt_count = pkt_count + 1
            return pkt_count
        except KeyboardInterrupt:
            self.udp_socket.close()
            print('User CTRL-C, stopping packet source')
            sys.exit(1)
        except:
            self.udp_socket.close()
            print("Got exception:", sys.exc_info()[0])
            raise

    def run(self) -> None:
        self.idle = True
        self.receiving = True
        print("Receival started.")
        while True:
            try:
                if self.receiving_from_mk5:
                    if not self.should_exit:
                        try:
                            recvd_bytes, recvd_from = self.udp_socket.recvfrom(65536)
                        except KeyboardInterrupt:
                            print("Keyboard interrupt raised.")
                            self.should_exit = True
                            self.receiving = False
                            self.idle = True
                            self.last_received_packet = ""
                            self.last_received_string = ""
                            self.ready_to_read = True
                            self.udp_socket.close()
                            break
                        except socket.timeout:
                            self.send_single_packet_to_router(curr_packet="ready", pkt_len=1500)
                            continue
                        if "break" in chr_long_string(bytearray(recvd_bytes)):
                            print("Receiving finished. Enter idle mode.")
                            self.last_received_string = "".join(self.received_packets)
                            self.idle = True
                            self.receiving = False
                            self.should_exit = False
                            self.ready_to_read = True
                        elif "begin" in chr_long_string(bytearray(recvd_bytes)):
                            print("Exiting idle mode. Renewing packet list...")
                            self.received_packets = []
                            self.idle = False
                            self.receiving = True
                            self.should_exit = False
                            self.ready_to_read = False
                        elif "exit" in chr_long_string(bytearray(recvd_bytes)):
                            self.should_exit = True
                            self.idle = True
                            self.receiving = False
                            self.ready_to_read = True
                        elif not self.idle:
                            self.last_received_packet = keep_useful_characters(strip_away_indicators(input_string=chr_long_string(bytearray(recvd_bytes)), indicator_string="ha"))
                            self.send_single_packet_to_router(curr_packet="ACK", pkt_len=1500)
                            print("Received something. Adding it to packets list. Received packet was of length: ", len(self.last_received_packet))
                            self.received_packets.append(self.last_received_packet)
                        elif self.idle:
                            self.send_single_packet_to_router(curr_packet="ready")
                            self.receiving = False
                            self.ready_to_read = True
                            self.receiving = False
                            self.idle = True
                    else:
                        print("Received ", len(self.received_packets), " many packets before exiting receival.")
                        print(self.last_received_string)
                        break
                else:
                    print("Received ", len(self.received_packets), " many packets before exiting.")
                    break
            except socket.timeout:
                self.send_single_packet_to_router(curr_packet="ready", pkt_len=1500)
                continue
            except KeyboardInterrupt:
                print("Keyboard interrupt raised. Exiting receival...")
                print("Received ", len(self.received_packets), " many packets before keyboard interrupt.")
                self.udp_socket.close()
                self.idle = True
                self.receiving = False
                self.should_exit = True
                self.udp_socket.close()
                break


class Smart_UDP_Receiver:
    def __init__(self, udp_socket_remote_address_arg: Tuple[str, int] = ("127.0.0.1", 5000), udp_socket_local_address_arg: Tuple[str, int] = ("127.0.0.1", 5250), udp_socket_timeout_arg: float = 5.0, udp_packet_buffer_size_arg: int = 1800, received_string_arg: str = "", received_strings_buffer_arg: List[str] = None, received_data_is_pickled_arg: bool = False, receiving_json_as_string_arg: bool = True, received_json_file_location_arg: str = "test_result.json", signals_address_arg: Tuple[str, int] = ("127.0.0.1", 10000), client_number_arg: int = 0, intra_packet_time_arg: float = 0.5, receiving_from_router_arg: bool = True):
        """
        Class designed to receive UDP packets in a smart fashion. This socket sends signal messages to only let the other side know that either the sockets timed out or something else happened. Signals can either directly be sent to the signal receiving address or sent to the remote address.

        :param udp_socket_remote_address_arg: This is the address that the socket will 'connect' to listen for UDP packets and send acknowledgements and disacknowledgements.
        :param udp_socket_local_address_arg: This is the address that the socket will listen from.
        :param udp_socket_timeout_arg: This is the amount of time needed to pass for a socket timeout. After this amount of time, the socket sends a disacknowledgement signal to the remote address.
        :param udp_packet_buffer_size_arg: The buffer size of the UDP socket used for listening to packets. This must be given in bytes and defaults to 1500, the MTU size for almost all sockets/interfaces.
        :param received_string_arg: This argument can be changed or used for debugging purposes. Not actively used in development or at run-time.
        :param received_strings_buffer_arg: This argument is also used for debugging purposes. Not actively used for develpoment or at run-time.
        :param received_data_is_pickled_arg: This argument is provided as a switch to let the socket know if it is meant to receive pickled data. If so, decoding the data will be made via the 'pickle' module.
        :param receiving_json_as_string_arg: This argument denotes if we are meant to receive JSON objects via strings or via bytes as in using FTP.
        :param received_json_file_location_arg: This argument is used to let the socket know the destination folder/file to save the received JSON objects into. Must end with the '.json' extension. No mechanism to check for file extension as of now.
        :param signals_address_arg: This is the address that the smart socket will send its ACK's, NACK's or other sorts of internal signals.
        """
        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.client_number = client_number_arg
        self.udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        self.udp_socket_remote_address = udp_socket_remote_address_arg
        self.udp_socket_local_address = udp_socket_local_address_arg
        self.signals_address = signals_address_arg
        self.udp_socket_timeout = udp_socket_timeout_arg
        self.udp_socket_buffer_size = udp_packet_buffer_size_arg
        self.received_string_buffer = []
        self.received_string = str()
        if received_string_arg == "":
            self.received_string = received_string_arg
        else:
            self.received_string = ""
        self.unpickled_data = Any  # Unpickled data could be of any type
        if received_strings_buffer_arg is None:
            received_strings_buffer_arg = []
            self.received_string_buffer = received_strings_buffer_arg
        else:
            self.received_string_buffer = received_strings_buffer_arg
        self.received_data_is_pickled = received_data_is_pickled_arg
        self.received_number = int(0)
        self.received_dict = dict()
        self.receiving_json_as_string = receiving_json_as_string_arg
        self.received_json_file_location = received_json_file_location_arg
        self.json_file_handle = open(self.received_json_file_location, "a+")
        self.json_file_handle.truncate(0)
        self.json_file_handle.close()
        self.curr_date_time = datetime.now()
        self.last_reliable_packet = ""
        self.received_packet_count = 0
        self.socket_timed_out = False
        self.nack_sent = False
        self.exiting = False
        self.acked_packet_index_list = list()
        self.intra_packet_time = intra_packet_time_arg
        logging.basicConfig(filename='receiver_address_' + str(self.udp_socket_local_address) + "time_" + self.curr_date_time.strftime("%d_%m_%Y %H_%M_%S") + '.log', filemode='a+', format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)
        logging.info("Object created at " + self.curr_date_time.strftime(curr_date_time_format))
        logging.info("Object properties are: " + str(vars(self)))
        self.receiving_from_router = receiving_from_router_arg
        self.ack_sent_or_not = False
        self.handshake_attempt_count = 0

    def __del__(self):
        self.json_file_handle.close()
        self.udp_socket.close()
        logging.info("Deleted all the file handles and also deleted the UDP receiver object at date and time: " + self.curr_date_time.now().strftime(curr_date_time_format))

    def send_single_packet_to_router(self, curr_packet: str = "", pkt_len: int = 1500):
        print("Test source to MK1 on %s:%d" % (self.udp_socket_remote_address[0], self.udp_socket_remote_address[1]))
        Target = (self.udp_socket_remote_address[0], self.udp_socket_remote_address[1])
        pkt_count = 0
        try:
            pktbuf = bytearray()
            some_string = "abcdefghijklmnopqrstuvwxyza|||" + curr_packet + "|||"  # For some reason, the router eats away 27 characters from the string to send. Possibly due to the UDP headers and fields overall taking up 24 bytes and 3 bytes used for some other stuff.
            for i in range(len(some_string)):
                pktbuf.append(ord(some_string[i]) & 0xff)
            for i in range(0, pkt_len - len(some_string)):
                pktbuf.append(0 & 0xff)
            print('Total packets transmitted: %d' % (pkt_count + 1))
            self.udp_socket.sendto(pktbuf, Target)
            pkt_count = pkt_count + 1
            return pkt_count
        except KeyboardInterrupt:
            self.udp_socket.close()
            print('User CTRL-C, stopping packet source')
            sys.exit(1)
        except:
            self.udp_socket.close()
            print("Got exception:", sys.exc_info()[0])
            raise

    def send_ACK_to_router(self, ack_index_arg: Optional[int] = None):
        ACK_index = 0
        if ack_index_arg is not None:
            ACK_index = ack_index_arg
        ack_sent = self.send_single_packet_to_router(curr_packet=("ACK " + str(ACK_index) + " "))
        self.ack_sent_or_not = bool(ack_sent)

    def receive_ACK_from_router(self):
        try:
            recvd_ack, recvd_from = self.udp_socket.recvfrom(65536, socket.MSG_WAITALL)
            recvd_ack_decoded = chr_long_string(bytearray(recvd_ack))
            if "ACK" in recvd_ack_decoded:
                print(recvd_ack_decoded)
                return True
            else:
                print("Got something else than ACK.")
                return False
        except KeyboardInterrupt:
            sys.exit(0)
        except socket.timeout:
            print("Did not get ACK, socket timed out.")

    def send_multiples_of_a_packet(self, curr_packet_arg: str = "", repeat_count: int = 3):
        for i in range(repeat_count):
            self.send_single_packet_to_router(curr_packet=curr_packet_arg)
            time.sleep(1.0)

    def receive_handshake(self):
        while True:
            try:
                self.udp_socket.settimeout(1.0)
                if not self.receiving_from_router:
                    message, recvd_from = self.udp_socket.recvfrom(65536)
                    self.handshake_attempt_count += 1
                    print("Message was: ", message.decode())
                    if message.decode() == "Coordinator initiated handshake" or "Coordinator initiated handshake" in message.decode():
                        print("Handshake initiation received from coordinator with IP address: ", recvd_from)
                        if self.udp_socket_remote_address == recvd_from:
                            pass
                        else:
                            print("Changing remote address to: ", recvd_from)
                            self.udp_socket_remote_address = recvd_from
                        self.udp_socket.sendto("OK".encode(), recvd_from)
                        break
                    else:
                        print("Got wrong handshake message.")
                        if self.handshake_attempt_count > 20:
                            print("Too many handshake attempts, exiting")
                            break
                        continue
                else:
                    self.udp_socket.settimeout(10.0)
                    message, recvd_from = self.udp_socket.recvfrom(65536)
                    print("Message was: ", chr_long_string(bytearray(message)))
                    if chr_long_string(bytearray(message)) == "Coordinator initiated handshake" or "Coordinator initiated handshake" in chr_long_string(bytearray(message)):
                        print("Handshake initiation received from coordinator with IP address: ", recvd_from)
                        self.send_multiples_of_a_packet(curr_packet_arg=" OK ",repeat_count=10)
                        break
                    else:
                        print("Got wrong handshake message.")
                        continue
                    pass
            except socket.timeout:
                print("Worker receiver handshake timed out.")
                self.handshake_attempt_count += 1
                if self.handshake_attempt_count > 20:
                    print("Too many handshake attemtps made, exiting")
                    break
                continue


    def bind_to_local_address(self):
        try:
            self.udp_socket.bind(self.udp_socket_local_address)
            print("Worker side receiver socket with client ID ", self.client_number, "will receive all sorts of signals and pakcets from address: ", self.udp_socket.getsockname())
        except OSError:
            for i in range(1025, 65536):
                try:
                    self.udp_socket.bind((self.udp_socket_local_address[0], i))
                    self.udp_socket_local_address = (self.udp_socket_local_address[0], i)
                    print("Receiver bound to address: ", self.udp_socket.getsockname(), " to receive information.")
                    break
                except OSError:
                    continue

    def change_udp_local_address(self, new_udp_socket_local_address_arg: Tuple[str, int]):
        self.udp_socket.close()
        logging.info("Deleted UDP socket object at date and time: " + self.curr_date_time.now().strftime(curr_date_time_format))
        self.udp_socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
        self.udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        self.udp_socket_local_address = new_udp_socket_local_address_arg
        self.udp_socket.bind(self.udp_socket_local_address)
        logging.info("Created new UDP socket at date and time: " + self.curr_date_time.now().strftime(curr_date_time_format) + " with porperties: " + str(vars(self.udp_socket)))

    def change_udp_remote_address(self, new_udp_socket_remote_address_arg: Tuple[str, int]):
        self.udp_socket_remote_address = new_udp_socket_remote_address_arg
        logging.info("Changed the remote address to: " + str(self.udp_socket_remote_address))

    def change_json_file_location(self, new_location: str = "new_json_file_location_for_thread_with_id0" + ".json"):
        if not self.json_file_handle.closed:
            self.json_file_handle.close()
        logging.info("Closed current JSON file at date and time: " + self.curr_date_time.now().strftime(curr_date_time_format))
        self.received_json_file_location = new_location
        self.json_file_handle = open(self.received_json_file_location, "a+")
        print("Changed JSON file location to: ", self.received_json_file_location)
        logging.info("Changed JSON file location at date and time: " + self.curr_date_time.now().strftime(curr_date_time_format) + "to: " + str(self.received_json_file_location))

    def receive_udp_packet(self):
        self.received_string_buffer = []
        print("Please send sockets to: ", self.udp_socket.getsockname())
        logging.info("Packet receival requested at date and time: " + self.curr_date_time.now().strftime(curr_date_time_format))
        print("Waiting...")
        self.udp_socket.settimeout(10.0)
        print("Worker receiver socket has timeout value of: ", self.udp_socket.timeout)
        while True:
            try:
                if not self.received_data_is_pickled:
                    if self.receiving_from_router:
                        recvd_message, recvd_from = self.udp_socket.recvfrom(65536)
                        print(chr_long_string(bytearray(recvd_message)))
                        pass

                    else:
                        try:
                            recvd_message, self.udp_socket_remote_address = self.udp_socket.recvfrom(65536, socket.MSG_WAITALL)
                            print("Received message from: ", self.udp_socket_remote_address)
                            self.received_string = recvd_message.decode()
                            self.udp_socket.sendto(("ACK " + str(self.received_packet_count)).encode(),
                                                   self.udp_socket_remote_address)
                            logging.info("Sent ACK signal to address " + str(
                                self.udp_socket_remote_address) + " at date and time " + self.curr_date_time.now().strftime(
                                curr_date_time_format))
                            self.received_packet_count += 1
                            self.socket_timed_out = False
                            self.nack_sent = False
                        except socket.timeout:
                            self.socket_timed_out = True
                            if not self.nack_sent:
                                for i in range(4):
                                    self.udp_socket.sendto("TIMEOUT\n".encode(), self.udp_socket_remote_address)
                                for i in range(4):
                                    self.udp_socket.sendto((self.last_reliable_packet + "\n").encode(), self.udp_socket_remote_address)
                            self.nack_sent = True
                            continue
                        except KeyboardInterrupt:
                            logging.info(
                                "Keyboaard exit requested at date and time: " + self.curr_date_time.now().strftime(
                                    curr_date_time_format))
                            self.nack_sent = False
                            self.socket_timed_out = False
                            sys.exit(0)
                        if self.received_string != "" and self.received_string != "\n":
                            if not self.socket_timed_out:
                                print(self.received_string)
                                if self.received_string == "Ready signal not received." or "Ready signal not received." in self.received_string:
                                    self.udp_socket.sendto("ready".encode(), self.udp_socket_remote_address)
                                elif "ready" in self.received_string or "break" in self.received_string or "ACK" in self.received_string:
                                    pass
                                else:
                                    self.acked_packet_index_list.append(self.received_packet_count - 1)
                                    print("Added index " + str(self.received_packet_count - 1) + " to ACK'ed indices.")
                                    pass
                            if self.received_string != "break" and not (
                                    "break" in self.received_string) and self.received_string != "exit" and not (
                                    "exit" in self.received_string):
                                self.received_string_buffer.append(keep_useful_characters(self.received_string))
                                self.last_reliable_packet = self.received_string
                                logging.info("Received packet of size: " + str(sys.getsizeof(
                                    self.received_string)) + " at time: " + self.curr_date_time.now().strftime(
                                    curr_date_time_format))
                            elif self.received_string == "exit" or "exit" in self.received_string:
                                self.exiting = True
                                break
                            elif self.received_string == "break" or "break" in self.received_string:
                                print("Receival finished.")
                                str_to_remove_cnt = self.received_string_buffer.count("Ready signal not received.")
                                for i in range(str_to_remove_cnt):
                                    self.received_string_buffer.remove("Ready signal not received.")
                                str_to_remove_cnt = self.received_string_buffer.count("Worker initiated handshake")
                                for i in range(str_to_remove_cnt):
                                    self.received_string_buffer.remove("Worker initiated handshake")
                                str_to_remove_cnt = self.received_string_buffer.count("Coordinator initiated handshake")
                                for i in range(str_to_remove_cnt):
                                    self.received_string_buffer.remove("Coordinator initiated handshake")
                                self.received_dict = json.loads("".join(self.received_string_buffer))
                                self.json_file_handle = open(self.received_json_file_location, "a+")
                                self.json_file_handle.truncate(0)
                                json.dump(self.received_dict, self.json_file_handle)
                                self.json_file_handle.close()
                                logging.info("Received a dictionary of size in bytes: " + str(sys.getsizeof(
                                    self.received_dict)) + " at date and time: " + self.curr_date_time.now().strftime(
                                    curr_date_time_format))
                                return eval("".join(self.received_string_buffer))
                        else:
                            self.udp_socket.sendto("ready".encode(), self.udp_socket_remote_address)
                            continue
                else:
                    self.unpickled_data = pickle.loads(self.udp_socket.recv(self.udp_socket_buffer_size))
                    if type(self.unpickled_data) is str:
                        self.received_string = self.unpickled_data
                        if self.received_string == "" or self.received_string == "\n":
                            self.udp_socket.sendto("ready".encode(), self.udp_socket_remote_address)
                        else:
                            print(self.received_string)
                            self.received_string_buffer.append(keep_useful_characters(self.received_string))
                    elif type(self.unpickled_data) is int:
                        self.received_number = self.unpickled_data
                    elif type(self.unpickled_data) is dict:
                        self.received_dict = self.unpickled_data
                    else:
                        print("Got data of type: ", type(self.unpickled_data), " if it is useful to you, reimplement this script according to your needs.")
            except TimeoutError:
                self.udp_socket.sendto("ready".encode(), self.udp_socket_remote_address)
                time.sleep(0.5)
            except socket.timeout:
                self.udp_socket.sendto("ready".encode(), self.udp_socket_remote_address)
                time.sleep(0.5)
            except KeyboardInterrupt:
                print("Keyboard interrupt raised.")
                logging.info("Keyboard interrupt raised at date and time: " + self.curr_date_time.now().strftime(curr_date_time_format))
                self.udp_socket.close()
                self.json_file_handle.close()
                sys.exit(0)


if __name__ == "__main__":
    thread_obj = Smart_UDP_Receiver_Thread()
    thread_obj.start()
    while True:
        try:
            thread_obj.join()
            print("Thread joined successfully.")
        except KeyboardInterrupt:
            print("Keyboard interrupt raised.")
            logging.info("Keyboard interrupt raised at date and time: " + datetime.now().strftime(curr_date_time_format))
            os.kill(os.getpid(), signal.SIGKILL)
            exit(0)
