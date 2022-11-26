import math
import socket
import textwrap
import time
from typing import Tuple
import sys
import os
import signal
from typing import List
import pickle
import numpy
import communication
import tensorflow as tf
import tensorflow._api.v2.compat.v1 as tf1_compat
import json
import io


# enable soft-wrapping in an editor please
init = 0
useful_characters = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "a", "b", "c", "d", "e", "f",  "g", "h", "i", "j", "k", "l", "m", "n", "o", "p",  "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "{", "}", "[", "]", ":", "-", ",", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", " ", "\"", "-", ".", ":"]


def ord_long_string(some_long_string: str = "") -> bytearray:
    returned_list = bytearray()
    for i in range(len(some_long_string)):
        returned_list.append(ord(some_long_string[i]) & 0xff)
    return returned_list


def chr_long_string(some_byte_array_arg: bytearray) -> str:
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


def download_model_json(ip: str = "127.0.0.1", port: int = 5000, time_out: int = 5000, fragment_size: int = 1500, window_size: int = 50, received_file_name: str = "received_unpickled_message_json_device0client0.json"):
    command_suffix = "--ip " + ip + " " + "--port " + str(port) + " " + "--time_out " + str(time_out) + " " + "--fragment_size " + str(fragment_size) + " " + "--window_size " + str(window_size) + " " + "--received_file_name " + received_file_name
    command = "python2.7 server_reliableUDP.py " + command_suffix
    while True:
        try:
            return_val = os.system(command)
            print(return_val)
            if return_val != 0:
                continue
            else:
                break
        except KeyboardInterrupt:
            os.kill(os.getpid(), signal.SIGUSR1)
            break


def upload_model_json(server_ip: str = "127.0.0.1", server_port: int = 1500, fragment_size: int = 1500, time_out: int = 50):
    command_suffix = "--server_ip " + server_ip + " " + "--server_port " + str(server_port) + " " + "--fragment_size " + str(fragment_size) + " " + "--time_out " + str(time_out)
    command = "python2.7 client_reliableUDP.py " + command_suffix
    while True:
        try:
            return_val = os.system(command)
            print(return_val)
            if return_val != 0:
                continue
            else:
                break
        except KeyboardInterrupt:
            os.kill(os.getpid(), signal.SIGUSR1)
            break


def close_file(file):
    file.close()


class TCP_2_UDP_Proxy:
    """
    This is an object/class solely written/created for encapsulating a TCP packet into a UDP packet. It is open to improvement/development. Refer to the documentation for the constructor for creating an object of this type.
    """
    def __init__(self, tcp_conn: socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM) = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM), tcp_sock: socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM) = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM), udp_sock: socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM) = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM), tcp_socket_local_address_arg: Tuple[str, int] = ("127.0.0.1", 5000), tcp_socket_remote_address_arg: Tuple[str, int] = ("127.0.0.1", 3001), udp_socket_remote_address_arg: Tuple[str, int] = ("127.0.0.1", 5100), udp_socket_local_address_arg: Tuple[str, int] = ("127.0.0.1", 1500), tcp_socket_received_strings_buffer_arg: List[str] = None, tcp_socket_timeout_arg: float = 10.0, udp_socket_timeout_arg: float = 10.0, data_is_pickled_arg: bool = True):
        """
            :parameter tcp_conn: This is the TCP connection object that the proxy will include within itself. It should be indirectly associated with the parameter "tcp_sock", via a call to either socket.create_connection() or socket.accept(). If not created externally, the proxy will create its own TCP socket and then connection for TCP packet receival.
            :param tcp_sock: This is the TCP socket object that the proxy will include within itself. If not created externally, the proxy will automatically create its own TCP socket.
            :param udp_sock: This is the UDP socket object that will be responsible for sending the UDP-encapsulated packets, after receiving the TCP packets. If not created externally, it will be created internally.
            :param tcp_socket_local_address_arg: This is the address that the TCP socket will listen from. External connections should be made targeting this address. External devices/objects should "connect to" this address.
            :param tcp_socket_remote_address_arg: This is the TCP address that the TCP socket would get its TCP connection and packets from. This address will be notified for readiness, so our local address or socket can receive TCP packets from it.
            :param udp_socket_remote_address_arg: This is the address that the UDP socket will send its encapsulated packets to.
            :param udp_socket_local_address_arg: This is the address that the UDP socket will listen incoming UDP packets
            :param tcp_socket_received_strings_buffer_arg: This is the buffer that holds the received TCP packets. Might be used for debugging or actual functionality within the code.
            :param tcp_socket_timeout_arg: This is the amount of time it takes for the TCP socket to time out. Currently, timeouts are handled automatically so it does not contribute much to functionality, however, the status quo for this argument might change in the future.
            :param udp_socket_timeout_arg: Similar to tcp_socket_timeout_arg, this argument also is used for timeout functionality. UDP socket timeouts are currently handled automatically as well by the class, however, the status quo for this argument might change as well.
            """
        self.packets = []
        self.max_udp_packet_size = 1500
        self.decoded_message_str = None
        self.decoded_message = None
        self.unpickled_message_json_file = open("init_unpickled_message_json_file.txt", "a+")
        self.received_string_file = open("init_received_string.txt", "a+")
        self.message_file = open("init_message.txt", "a+")
        self.unpickled_message_file = open("init_unpickled_message.txt", "a+")
        self.tcp_connection = tcp_conn
        self.tcp_socket = tcp_sock
        self.udp_socket = udp_sock
        self.tcp_socket_local_address = tcp_socket_local_address_arg
        self.tcp_socket_remote_address = tcp_socket_remote_address_arg
        self.udp_socket_remote_address = udp_socket_remote_address_arg
        self.udp_socket_local_address = udp_socket_local_address_arg
        tcp_socket_received_strings_buffer_arg = []
        self.tcp_socket_received_strings_buffer = tcp_socket_received_strings_buffer_arg
        self.received_data = bytearray("", "utf-8")
        self.message = bytearray()
        self.received_string = ""
        self.unpickled_message = dict()
        self.tcp_socket_timeout = tcp_socket_timeout_arg
        self.udp_socket_timeout = udp_socket_timeout_arg
        self.data_is_pickled = data_is_pickled_arg
        self.data_from_init = True
        self.message_file_location = "received_message_bytearray_before_pickle_loads.txt"
        self.unpickled_message_file_location = "received_message_after_pickle_loads.pkl"
        self.received_string_file_location = "received_string.txt"
        self.unpickled_message_json_file.close()
        self.received_string_file.close()
        self.message_file.close()
        self.unpickled_message_file.close()
        global init
        self.unpickled_message_json_file_location = "unpickled_message_json_device" + str(init)
        init += 1
        self.unpickled_message_file_stats = None
        self.message_file_stats = None
        self.received_string_file_stats = None
        self.unpickled_message_json = {"weights": list(), "biases": list()}
        self.unpickled_message_keys = list(self.unpickled_message.keys())
        self.unpickled_message_values = list(self.unpickled_message.values())
        self.model_weights = list(tf.Variable(initial_value=[]))
        self.model_biases = list(tf.Variable(initial_value=[]))
        self.model_weights_numerical_values = list()
        self.model_biases_numerical_values = list()
        try:
            self.tcp_connection = socket.create_connection(self.tcp_socket_remote_address)
            # while True:
            #     if self.data_is_pickled:
            #         # Receive message from the coordinator
            #         self.message = communication.receive_data(self.tcp_connection)
            #         print("Received message has type (type(self.message)): ", type(self.message))
            #         print("Received message of size (sys.getsizeof(self.message)): ", sys.getsizeof(self.message))
            #         print("Receied message of size (len(self.message))", len(self.message))
            #
            #         # Write the received message (bytearray) into a file for later inspection
            #         self.open_message_file()
            #         self.message_file.write(self.message)
            #         self.message_file.close()
            #         self.message_file_stats = os.stat(self.message_file_location)
            #
            #         # Unpickle the received message (which is in bytearray form)
            #         self.unpickled_message = pickle.loads(self.message)
            #         print("Received unpickled message has type (type(self.unpickled_message)): ",
            #               type(self.unpickled_message))
            #         print("Received unpickled message of size (sys.getsizeof): ", sys.getsizeof(self.unpickled_message))
            #         if type(self.unpickled_message) is dict:
            #             print("Received unpickled message of size (len): ", len(self.unpickled_message))
            #             # Convert the unpickled message (the model) into JSON object and save it.
            #             self.open_unpickled_message_json_file()
            #             json.dump(self.unpickled_message_json, self.unpickled_message_json_file)
            #         elif type(self.unpickled_message) is int:
            #             print("Received unpickled message is of type int and the integer value stored is: ", self.unpickled_message)
            #             self.unpickled_message_json_file_location = self.unpickled_message_json_file_location + "client" + str(
            #                 self.unpickled_message) + ".json"
            #         print("Received unpickled object has properties: ", dir(self.unpickled_message))
            #
            #         # Write the unpickled message into a file for later inspection
            #         self.open_unpickled_message_file()
            #         pickle.dump(self.unpickled_message, self.unpickled_message_file)
            #         self.unpickled_message_file.close()
            #         self.unpickled_message_file_stats = os.stat(self.unpickled_message_file_location)
            #         print("Unpickled file size is: ", self.unpickled_message_file_stats.st_size)
            #
            #         # Convert the unpickled message into string form
            #         self.received_string = str(self.unpickled_message)
            #         print("Received string of size (sys.getsizeof): ", sys.getsizeof(self.received_string))
            #         print("Received string of size (len): ", len(self.received_string))
            #         print("Received string is of type (type(self.received_string)):", type(self.received_string))
            #         self.open_received_string_file()
            #         self.received_string_file.write(self.received_string)
            #         self.received_string_file.close()
            #         self.received_string_file_stats = os.stat(self.received_string_file_location)
            #         print("Received string file size is: ", self.received_string_file_stats.st_size)
            #         break
            #     else:
            #         self.received_data = self.tcp_connection.recv(1800)
            #         self.received_string = self.received_data.decode()
            #
            #     if self.received_string == "exit" or self.received_string == "exit\n":
            #         print("On init, received exit signal.")
            #         break
            #     elif self.received_string == "" or self.received_string == "\n":
            #         continue
            #     else:
            #         print("On init, received")
            #         print("Received unpickled data: ", self.unpickled_message)
            #         print("Received pure string data of: ", self.received_string)
            #         break
            # self.data_from_init = True
        except ConnectionRefusedError:
            while True:
                try:
                    self.tcp_connection = socket.create_connection(self.tcp_socket_remote_address)
                    # while True:
                    #     self.received_data = self.tcp_connection.recv(1800)
                    #     if self.data_is_pickled:
                    #         self.message = communication.receive_data(self.tcp_connection)
                    #         self.unpickled_message = pickle.loads(self.message)
                    #         self.received_string = str(self.unpickled_message)
                    #     else:
                    #         self.received_string = self.received_data.decode()
                    #
                    #     if self.received_string == "exit" or self.received_string == "exit\n":
                    #         break
                    #     elif self.received_string == "" or self.received_string == "\n":
                    #         continue
                    #     else:
                    #         print("Received unpickled data: ", self.unpickled_message)
                    #         print("Received pure string data of: ", self.received_string)
                    #         continue
                    break
                except ConnectionRefusedError:
                    continue
        print("Please connect to TCP address ", self.tcp_socket_local_address, " for sending your demanded TCP packets.")
        print("This socket is meant to send its UDP packets towards UDP socket with address ", self.udp_socket_remote_address, " from UDP socket address ", self.udp_socket_local_address)

    def send_single_packet_to_router(self, curr_packet: str = "", pkt_len: int = 1600):
        print("Test source to MK1 on %s:%d" % (self.udp_socket_remote_address[0], self.udp_socket_remote_address[1]))
        pkt_count = 0
        # Open the socket to communicate with the mk1
        Target = self.udp_socket_remote_address  # the target address and port
        self.udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        try:
            pktbuf = bytearray()
            some_string = "|||||||||||||||||||||||||||  ^||" + curr_packet + "^||^||"  # For some reason, the router eats away 27 characters from the string to send. Possibly due to the UDP headers and fields overall taking up 24 bytes and 3 bytes used for some other stuff.
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
            return pkt_count
        except KeyboardInterrupt:
            self.udp_socket.close()
            print('User CTRL-C, stopping packet source')
            sys.exit(1)
        except:
            print("Got exception:", sys.exc_info()[0])
            raise

    def close_unpickled_message_file(self):
        self.unpickled_message_file.close()

    def close_message_file(self):
        self.message_file.close()

    def close_received_string_file(self):
        self.received_string_file.close()

    def open_unpickled_message_file(self):
        self.unpickled_message_file = open(self.unpickled_message_file_location, "ab+")

    def open_message_file(self):
        self.message_file = open(self.message_file_location, "ab+")

    def open_received_string_file(self):
        self.received_string_file = open(self.received_string_file_location, "a+")

    def open_unpickled_message_json_file(self):
        self.unpickled_message_json_file = open(self.unpickled_message_json_file_location, "a+")

    def open_and_clear_unpickled_message_json_file(self):
        self.unpickled_message_json_file = open(self.unpickled_message_json_file_location, "a+")
        self.unpickled_message_json_file.truncate(0)

    def open_and_clear_unpickled_message_file(self):
        self.open_unpickled_message_file()
        self.unpickled_message_file.truncate(0)

    def open_and_clear_message_file(self):
        self.open_message_file()
        self.message_file.truncate(0)

    def open_and_clear_received_string_file(self):
        self.open_received_string_file()
        self.received_string_file.truncate(0)

    def configure_connections_and_sockets(self):
        while True:
            try:
                self.tcp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self.tcp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
                self.tcp_socket.bind(self.tcp_socket_local_address)
                self.tcp_socket.listen(1)
                self.tcp_connection, self.tcp_socket_remote_address = self.tcp_socket.accept()
                print("After configuration of sockets (configure_connections_and_sockets), TCP socket's local address became ", self.tcp_socket_local_address)
                print("After configuration of sockets (configure_connections_and_sockets), TCP socket's remote address became ", self.tcp_socket_remote_address)
                print("After configuration of sockets (configure_connections_and_sockets), TCP connection became ", self.tcp_connection)
                print("After configuration of sockets (configure_connections_and_sockets), UDP address didn't change ")
                break
            except socket.timeout:
                continue
            except OSError:
                continue

    def change_tcp_connection_external(self, new_tcp_conn: socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM), new_tcp_sock: socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)):
        """
        Currently used purely for debugging purposes, might be used externally by some other entity or internally by this class for other purposes. The reason it is called change_tcp_connection_external is that the TCP connection and socket must be made externally by some other entity.

        :param new_tcp_conn:
        :param new_tcp_sock:
        :return: None
        """
        self.tcp_connection = new_tcp_conn
        self.tcp_socket = new_tcp_sock

    def change_tcp_connection_internal(self, new_tcp_remote_address: Tuple[str, int]):
        """
        This function, unlike its external counterpart or bredrin, establishes the TCP connection internally, given the remote address to connect to. Everytihng is taken care of by the class and the function in terms of connection. No need to explain further why it is called the way it is.

        :param new_tcp_remote_address: The new remote address to connect to.
        :return: None
        """

        # Shut the unneeded connection down, without releasing the resources associated with the socket
        self.tcp_connection.shutdown(socket.SHUT_RDWR)
        self.tcp_socket.shutdown(socket.SHUT_RDWR)
        self.tcp_socket.close()

        # Establish the new connection
        self.tcp_socket_remote_address = new_tcp_remote_address
        self.tcp_socket = socket.socket()
        try:
            self.tcp_socket.connect(("127.0.0.1", 5000))
            while True:
                received_data = self.tcp_socket.recv(1800).decode()
                if received_data == "exit" or received_data == "exit\n":
                    break
                elif received_data == "" or received_data == "\n":
                    continue
                else:
                    print(received_data)
                    continue
        except ConnectionRefusedError:
            while True:
                try:
                    self.tcp_socket.connect(("127.0.0.1", 5000))
                    print("Connection reestablished")
                    while True:
                        received_data = self.tcp_socket.recv(1800).decode()
                        if received_data == "exit" or received_data == "exit\n":
                            break
                        elif received_data == "" or received_data == "\n":
                            continue
                        else:
                            print(received_data)
                            continue
                    break
                except ConnectionRefusedError:
                    continue

    def toJSON(self):
        return json.dumps(self.unpickled_message_json, default=lambda o: o.__dict__,
                          sort_keys=True, indent=4)

    def rcv_tcp_and_send_udp(self):
        # UDP stuff and bindings
        print("UDP target IP: ", self.udp_socket_remote_address[0])
        print("UDP target port: ", self.udp_socket_remote_address[1])
        print("TCP socket got connection from address: ", self.tcp_connection.getpeername())
        udp_socket_max_size = 1800
        while True:
            try:
                if self.data_is_pickled:
                    if self.data_from_init:
                        # Send initial message via UDP
                        self.udp_socket.sendto(str(self.unpickled_message).encode(), self.udp_socket_remote_address)
                        self.data_from_init = False
                        pass
                    else:
                        # Receive message from the coordinator
                        self.message = communication.receive_data(self.tcp_connection)
                        print("Received message has type (type(self.message)): ", type(self.message))
                        print("Received message of size (sys.getsizeof(self.message)): ", sys.getsizeof(self.message))
                        print("Receied message of size (len(self.message))", len(self.message))
                        self.open_message_file()
                        # Write the received message (bytearray) into a file for later inspection
                        self.message_file.write(self.message)
                        self.message_file.close()
                        self.message_file_stats = os.stat(self.message_file_location)

                        # Unpickle the received message (which is in bytearray form)
                        self.unpickled_message = pickle.loads(self.message)
                        print("Received unpickled message has type (type(self.unpickled_message)): ",
                              type(self.unpickled_message))
                        print("Received unpickled message of size (sys.getsizeof): ",
                              sys.getsizeof(self.unpickled_message))
                        if type(self.unpickled_message) is dict:
                            print("Received unpickled message of size (len): ", len(self.unpickled_message))
                            # Convert the unpickled message (the model) into JSON object and save it.
                            self.unpickled_message = dict(self.unpickled_message)
                            self.unpickled_message_keys = list(self.unpickled_message.keys())
                            self.unpickled_message_values = list(self.unpickled_message.values())
                            for i in range(len(self.unpickled_message_keys)):
                                print("Key: ", self.unpickled_message_keys[i], " Value type: ", type(self.unpickled_message_values[i]), " Value: ", self.unpickled_message_values[i])
                                if self.unpickled_message_keys[i] == "weights":
                                    self.model_weights.append(self.unpickled_message_values[i])
                                    print("Model weights variable is of type: ", type(self.model_weights))
                                    print("Model weights variable includes variables of type ",
                                          type(self.model_weights[-1]), " within")
                                    for k in range(len(self.model_weights[-1])):
                                        self.model_weights_numerical_values.append(self.model_weights[-1][k].read_value())
                                        self.unpickled_message_json["weights"].append(self.model_weights[-1][k].read_value().numpy().tolist())
                                elif self.unpickled_message_keys[i] == "biases":
                                    self.model_biases.append(self.unpickled_message_values[i])
                                    for k in range(len(self.model_biases[-1])):
                                        self.model_biases_numerical_values.append(self.model_biases[-1][k].read_value())
                                        self.unpickled_message_json["biases"].append(self.model_biases[-1][k].read_value().numpy().tolist())
                                else:
                                    print("The key is something else")

                            # self.toJSON()
                            self.open_and_clear_unpickled_message_json_file()
                            json.dump(self.unpickled_message_json, self.unpickled_message_json_file)
                            self.unpickled_message_json_file.close()

                        elif type(self.unpickled_message) is int:
                            print("Received unpickled message is of type int and the integer value stored is: ",
                                  self.unpickled_message)
                            self.unpickled_message_json_file_location = self.unpickled_message_json_file_location + "client" + str(self.unpickled_message) + ".json"
                        print("Received unpickled object has properties: ", dir(self.unpickled_message))
                        # Write the unpickled message into a file for later inspection
                        self.open_unpickled_message_file()
                        pickle.dump(self.unpickled_message, self.unpickled_message_file)
                        self.unpickled_message_file.close()
                        self.unpickled_message_file_stats = os.stat(self.unpickled_message_file_location)
                        print("Unpickled file size is: ", self.unpickled_message_file_stats.st_size)

                        # Convert the unpickled message into string form
                        self.received_string = str(self.unpickled_message)
                        print("Received string of size (sys.getsizeof): ", sys.getsizeof(self.received_string))
                        print("Received string of size (len): ", len(self.received_string))
                        print("Received string is of type (type(self.received_string)):", type(self.received_string))
                        self.open_received_string_file()
                        self.received_string_file.write(self.received_string)
                        self.received_string_file.close()
                        self.received_string_file_stats = os.stat(self.received_string_file_location)
                        print("Received string file size is: ", self.received_string_file_stats.st_size)
                        # UDP sending code here
                        os.system("python2.7 client_reliableUDP.py")

                else:
                    self.received_data = self.tcp_connection.recv(65536)
                    self.received_string = self.received_data.decode()
                    if self.received_string == "" or self.received_string == "\n":
                        print("Nothing received, waiting for new packets...")
                        self.configure_connections_and_sockets()
                        time.sleep(1)
                        continue
                    else:
                        model_data_list = textwrap.wrap(self.received_data.decode(), udp_socket_max_size)
                        self.tcp_socket_received_strings_buffer.extend(model_data_list)
                        print(model_data_list)
                        chunk_length_list = []
                        for i in range(len(model_data_list)):
                            self.udp_socket.sendto(model_data_list[i].encode(), self.udp_socket_remote_address)
                            chunk_length_list.append(str(i) + ": " + str(len(model_data_list[i])) + " ")
                        print("Sent data")
                        print("Data was of length: ", len(self.received_data))
                        print("Data took ", math.ceil(len(self.received_data) / udp_socket_max_size), " many chunks")
                        print("Data included chunks of size: ", chunk_length_list)
                        continue
            except socket.timeout:
                print("Socket timed out. ")
                self.udp_socket.close()
                break
            except KeyboardInterrupt:
                self.tcp_connection.close()
                break

    def recv_any_byte_and_send_it(self, mk5_used: bool = True):
        print("Receiving to TCP address ", self.tcp_connection.getsockname(), " sending to UDP address ", self.udp_socket_remote_address)
        while True:
            try:
                self.message = communication.receive_data(self.tcp_connection)
                self.decoded_message = pickle.loads(self.message)
                print(self.decoded_message)
                self.decoded_message_str = str(self.decoded_message)
                self.packets = []
                if mk5_used:
                    if len(self.decoded_message_str) > self.max_udp_packet_size:
                        self.packets = textwrap.wrap(self.decoded_message_str, self.max_udp_packet_size)
                        index = 0
                        while index < len(self.packets):
                            self.send_single_packet_to_router(curr_packet=self.packets[index],
                                                              pkt_len=len(self.packets[index]) + 100)
                            index += 1
                            time.sleep(0.2)
                        self.send_single_packet_to_router(curr_packet="DONE")
                        print(self.packets)
                    else:
                        self.send_single_packet_to_router(curr_packet=self.decoded_message_str, pkt_len=len(self.decoded_message_str) + 100)
                        self.send_single_packet_to_router(curr_packet="DONE")
                        print("Sent: ", self.decoded_message_str)
                else:
                    if len(self.decoded_message_str) > self.max_udp_packet_size:
                        self.packets = textwrap.wrap(self.decoded_message_str, self.max_udp_packet_size)
                        for i in range(len(self.packets)):
                            self.udp_socket.sendto(self.packets[i].encode(), self.udp_socket_remote_address)
                        self.udp_socket.sendto("DONE".encode(), self.udp_socket_remote_address)
                    else:
                        self.udp_socket.sendto(self.decoded_message.encode(), self.udp_socket_remote_address)
                        self.udp_socket.sendto("DONE".encode(), self.udp_socket_remote_address)
            except KeyboardInterrupt:
                sys.exit(0)
            except:
                print(sys.exc_info())
                continue


if __name__ == "__main__":
    test_object = TCP_2_UDP_Proxy(tcp_socket_local_address_arg=("127.0.0.1", 2250), tcp_socket_remote_address_arg=("127.0.0.1", 3001), udp_socket_remote_address_arg=("192.168.1.41", 4000), udp_socket_local_address_arg=("192.168.1.57", 2000))
    test_object.recv_any_byte_and_send_it()
    print("Done")
    pass
