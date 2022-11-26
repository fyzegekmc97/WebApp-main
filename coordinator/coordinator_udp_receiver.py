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
import numpy as np
import tensorflow as tf

total_clients = 2
curr_date_time_format = "%d/%m/%Y, %H:%M:%S"
useful_characters = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "a", "b", "c", "d", "e", "f",  "g", "h", "i", "j", "k", "l", "m", "n", "o", "p",  "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "{", "}", "[", "]", ":", "-", ",", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", " ", "\"", "-", ".", ":"]
numberOfNeurons = [784, 128, 64, 2]  # length of this array must be numberOfLayers + 1
numberOfLayers = 3  # Total number of layers expected within the neural network


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


class AlwaysOpenUDP_Receiver(Thread):
    def __init__(self, local_ip_address: str = "192.168.1.57", remote_ip_address: str = "192.168.1.31", local_port: int = 4003, remote_port: int = 4000):
        Thread.__init__(self)
        self.udp_socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
        self.local_ip = local_ip_address
        self.local_port = local_port
        self.remote_ip = remote_ip_address
        self.remote_port = remote_port
        self.udp_socket.bind((self.local_ip, self.local_port))
        self.received_packets = []
        self.udp_socket.settimeout(1.0)
        self.receiving_from_mk5 = True
        self.should_exit = False
        self.idle = True
        self.receiving = False
        self.last_received_string = ""
        self.ready_to_read = False

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
                            print()
                            break
                        if "break" in chr_long_string(bytearray(recvd_bytes)):
                            print("Receiving finished. Enter idle mode.")
                            self.last_received_string = "".join(self.received_packets)
                            self.idle = True
                            self.ready_to_read = True
                            self.should_exit = False
                        elif "begin" in chr_long_string(bytearray(recvd_bytes)):
                            print("Exiting idle mode. Renewing packet list...")
                            self.received_packets = []
                            self.idle = False
                            self.ready_to_read = False
                            self.should_exit = False
                        elif "exit" in chr_long_string(bytearray(recvd_bytes)):
                            self.should_exit = True
                            self.idle = True
                            self.receiving = False
                            self.ready_to_read = False
                        elif not self.idle:
                            print("Received something. Adding it to packets list...")
                            self.ready_to_read = False
                            self.received_packets.append(keep_useful_characters(strip_away_indicators(input_string=chr_long_string(bytearray(recvd_bytes)), indicator_string="ha")))
                        elif self.idle:
                            self.send_single_packet_to_router(curr_packet="ready")
                            self.receiving = False
                            self.ready_to_read = True
                            self.receiving = False
                            self.idle = True
                    else:
                        print("Received ", len(self.received_packets), " many packets before exiting receival.")
                        print(self.last_received_string)
                        self.idle = True
                        self.ready_to_read = True
                        self.receiving = False
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
                break


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


class Smart_UDP_Receiver_Thread(Thread):
    def __init__(self):
        Thread.__init__(self)
        self.receiver = Smart_UDP_Receiver()

    def run(self) -> None:
        self.receiver.receive_udp_packet()
        print("One receival finished.")


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


class Smart_UDP_Receiver:
    def __init__(self, udp_socket_arg: socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM) = None, udp_socket_remote_address_arg: Tuple[str, int] = ("127.0.0.1", 5000), udp_socket_local_address_arg: Tuple[str, int] = ("127.0.0.1", 5250), udp_packet_buffer_size_arg: int = 1800, received_string_arg: str = "", received_strings_buffer_arg: List[str] = None, received_data_is_pickled_arg: bool = False, receiving_json_as_string_arg: bool = True, received_json_file_location_arg: str = "test_result.json", signals_address_arg: Tuple[str, int] = ("127.0.0.1", 10000), client_number_arg: int = 0, intra_packet_time_arg: float = 0.05, sending_to_router_arg: bool = True):
        """
        Class designed to receive UDP packets in a smart fashion. This socket sends signal messages to only let the other side know that either the sockets timed out or something else happened. Signals can either directly be sent to the signal receiving address or sent to the remote address.

        :param udp_socket_arg: Externally made socket for receival of UDP packets, optional to provide externally. If not provided externally, it will be internally made.
        :param udp_socket_remote_address_arg: This is the address that the socket will 'connect' to listen for UDP packets and send acknowledgements and disacknowledgements.
        :param udp_socket_local_address_arg: This is the address that the socket will listen from.
        :param udp_packet_buffer_size_arg: The buffer size of the UDP socket used for listening to packets. This must be given in bytes and defaults to 1500, the MTU size for almost all sockets/interfaces.
        :param received_string_arg: This argument can be changed or used for debugging purposes. Not actively used in development or at run-time.
        :param received_strings_buffer_arg: This argument is also used for debugging purposes. Not actively used for develpoment or at run-time.
        :param received_data_is_pickled_arg: This argument is provided as a switch to let the socket know if it is meant to receive pickled data. If so, decoding the data will be made via the 'pickle' module.
        :param receiving_json_as_string_arg: This argument denotes if we are meant to receive JSON objects via strings or via bytes as in using FTP.
        :param received_json_file_location_arg: This argument is used to let the socket know the destination folder/file to save the received JSON objects into. Must end with the '.json' extension. No mechanism to check for file extension as of now.
        :param signals_address_arg: This is the address that the smart socket will send its ACK's, NACK's or other sorts of internal signals.
        """
        if udp_socket_arg is not None:
            self.udp_socket = udp_socket_arg
        else:
            self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.client_number = client_number_arg
        self.udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        self.udp_socket_remote_address = udp_socket_remote_address_arg
        self.udp_socket_local_address = udp_socket_local_address_arg
        self.signals_address = signals_address_arg
        self.udp_socket_buffer_size = udp_packet_buffer_size_arg
        self.received_string_buffer = []
        self.received_string = str()
        self.exiting = False
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
        self.udp_socket.settimeout(1.0)
        self.acked_packet_index_list = list()
        logging.basicConfig(filename='receiver_address_' + str(self.udp_socket_local_address) + "time_" + self.curr_date_time.strftime("%d_%m_%Y %H_%M_%S") + '.log', filemode='a+', format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)
        logging.info("Object created at " + self.curr_date_time.strftime(curr_date_time_format))
        logging.info("Object properties are: " + str(vars(self)))
        self.intra_packet_time = intra_packet_time_arg
        self.sending_to_router = sending_to_router_arg
        self.ack_sent_or_not = False

    def __del__(self):
        self.json_file_handle.close()
        self.udp_socket.close()
        logging.info("Deleted all the file handles and also deleted the UDP receiver object at date and time: " + self.curr_date_time.now().strftime(curr_date_time_format))

    def send_single_packet_to_router(self, curr_packet: str = "", pkt_len: int = 1900):
        print("Test source to MK1 on %s:%d" % (self.udp_socket_remote_address[0], self.udp_socket_remote_address[1]))
        # Open the socket to communicate with the mk1
        Target = (self.udp_socket_remote_address[0], self.udp_socket_remote_address[1])  # the target address and port
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
            print("Transmitted %d packets" % pkt_count)
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
            recvd_ack, recvd_from = self.udp_socket.recvfrom(1500)
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

    def receive_handshake(self):
        while True:
            try:
                if not self.sending_to_router:
                    print("Coordinator r3eceiver started receiving handshake from remote device...")
                    message, recvd_from = self.udp_socket.recvfrom(65536)
                    print("Message was: ", message.decode())
                    if message.decode() == "Worker initiated handshake" or "Worker initiated handshake" in message.decode():
                        print("Handshake initiation received from worker with IP address: ", recvd_from)
                        if self.udp_socket_remote_address == recvd_from:
                            pass
                        else:
                            print("Changing remote address to: ", recvd_from)
                            self.udp_socket_remote_address = recvd_from
                        self.udp_socket.sendto("OK".encode(), recvd_from)
                        break
                    else:
                        print("Got wrong handshake message.")
                        continue
                else:
                    print("Coordinator receiver started receiving handshake from router...")
                    message, recvd_from = self.udp_socket.recvfrom(65536)
                    print("Message was: ", chr_long_string(bytearray(message)))
                    if chr_long_string(bytearray(message)) == "Worker initiated handshake" or "Worker initiated handshake" in chr_long_string(bytearray(message)):
                        print("Handshake initiation received from worker with IP address: ", recvd_from)
                        self.send_multiples_of_a_packet(curr_packet_arg=" OK ", repeat_count=3)
                        break
                    else:
                        print("Got wrong handshake message.")
                        continue
                    pass
            except socket.timeout:
                print("Coordinator receiver handshake timed out.")
                continue

    def bind_to_local_address(self):
        try:
            self.udp_socket.bind(self.udp_socket_local_address)
            print("Parameter server receiver socket for client ID", self.client_number, " will get its ACK's, NACK's and general packages from address: ",  self.udp_socket.getsockname())
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

    def send_multiple_signals_to_socket_in_a_second(self, curr_packet: str = "",  pkt_rate: int = 3, pkt_len: int = 1500, num_pkts: int = 15):
        """
            Test packet source routine, which generates UDP packets to be forwarded by
            the fwd_wsmp_forward_tx utility.

            This requires that the utility 'fwd_wsmp_forward_tx' is running on the
            target radio.

            Arguments:
            mk1_addr - the IP Address of the MK1 Radio used to transmit packet
                       e.g. '192.168.227.227', or '255.255.255.255'
            mk1_port - the port of the MK1 Radio used to transmit packet
                       e.g. 4040
            pkt_rate - the number of packets per second to generate
            pkt_len  - Number of bytes in the packet
            num_pkts - Number of packets to generate (-1 means forever)
            """
        print("Test source to MK1 on %s:%d" % (self.udp_socket_remote_address[0], self.udp_socket_remote_address[1]))
        self.udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        Target = self.udp_socket_remote_address
        pkt_count = 0
        try:
            # Main loop
            while (pkt_count < num_pkts) or (num_pkts == -1):
                pktbuf = bytearray()
                some_string = "hahahahahahahahahahahahahahahaha" + curr_packet + "hahahaha"  # For some reason, the router eats away 27 characters from the string to send. Possibly due to the UDP headers and fields overall taking up 24 bytes and 3 bytes used for some other stuff.
                for i in range(len(some_string)):
                    pktbuf.append(ord(some_string[i]) & 0xff)
                for i in range(0, pkt_len - len(some_string)):
                    pktbuf.append(0 & 0xff)

                print('Total packets transmitted: %d' % (pkt_count + 1))
                self.udp_socket.sendto(pktbuf, Target)
                pkt_count = pkt_count + 1
            print("Transmitted %d packets" % pkt_count)
            return pkt_count

        except KeyboardInterrupt:
            print('User CTRL-C, stopping packet source')
            sys.exit(1)
        except:
            print("Got exception:", sys.exc_info()[0])
            raise

    def change_json_file_location(self, new_location: str = "new_json_file_location_for_thread_with_id" + str(threading.get_native_id()) + ".json"):
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
        print("")
        self.udp_socket.close()
        self.udp_socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
        print("Rebinding...")
        while True:
            try:
                self.udp_socket.bind(self.udp_socket_local_address)
                break
            except:
                continue
        if self.received_data_is_pickled:
            while True:
                try:
                    # Unpickle pickled data
                    self.unpickled_data = pickle.loads(self.udp_socket.recv(65536))
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
                    if self.sending_to_router:
                        self.send_single_packet_to_router(curr_packet="ready", pkt_len=1500)
                        pass
                    else:
                        self.udp_socket.sendto("ready".encode(), self.udp_socket_remote_address)
                    time.sleep(0.5)
                except KeyboardInterrupt:
                    print("Keyboard interrupt raised.")
                    logging.info("Keyboard interrupt raised at date and time: " + self.curr_date_time.now().strftime(curr_date_time_format))
                    self.udp_socket.close()
                    self.json_file_handle.close()
                    sys.exit(0)
        elif self.sending_to_router:
            while True:
                recvd_bytes, recvd_from = self.udp_socket.recvfrom(2000)
                print(recvd_bytes)

    def close_socket(self):
        self.udp_socket.close()


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
