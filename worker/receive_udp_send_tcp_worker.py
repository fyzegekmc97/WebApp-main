import socket
from datetime import datetime
import sys
import os
import signal
from typing import Tuple
import json
import communication


useful_characters = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "a", "b", "c", "d", "e", "f",  "g", "h", "i", "j", "k", "l", "m", "n", "o", "p",  "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "{", "}", "[", "]", ":", "-", ",", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", " ", "\"", "-", ".", ":"]


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


def download_model_json(ip: str = "127.0.0.1", port: int = 5000, time_out: float = 5000, fragment_size: int = 1500, window_size: int = 50, received_file_name: str = "received_unpickled_message_json_device0client0.json"):
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


def upload_model_json(server_ip: str = "127.0.0.1", server_port: int = 1500, fragment_size: int = 1500, window_size: int = 50, time_out: int = 50, file_name: str = "unpickled_message_json_device0client0.json"):
    command_suffix = "--server_ip " + server_ip + " " + "--server_port " + str(server_port) + " " + "--fragment_size " + str(fragment_size) + " " + "--window_size " + str(window_size) + " " + "--time_out " + str(time_out) + " " + "--file_name " + file_name
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


class UDP_2_TCP_Proxy:
    """
    This class is a proxy class for receiving UDP packets and then turning them into TCP packets. Likewise, one can also turn UDP packets into TCP packets. Intended for use with threads, continuously listening to UDP packets at some port. Also, a function is provided for smart reconnection to the same TCP socket, in case the connection drops with the TCP socket due to some unprecendented reason. Functions are provided as well for changing the TCP connection, in case something happens, or in case the future users need to change the TCP connection at wish at runtime. Refer to the constructor documentation for creating an object of this type.
    """
    def __init__(self, tcp_conn: socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM) = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM), tcp_sock: socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM) = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM), udp_sock: socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM) = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM), tcp_connaction_local_address_arg: Tuple[str, int] = ("127.0.0.1", 9000),tcp_connection_remote_address_arg: Tuple[str, int] = ("127.0.0.1", 3001), udp_socket_remote_address_arg: Tuple[str, int] = ("127.0.0.1", 5250), udp_socket_local_address_arg: Tuple[str, int] = ("127.0.0.1", 10000), udp_socket_timeout_arg: float = 3.0, tcp_socket_timeout_arg: float = 3.0, client_number_arg: int = 0, receiving_json_as_file_arg: bool = False, device_number_arg: int = 0, tcp_socket_expects_str_arg: bool = True, receiving_json_as_string_arg: bool = True, json_file_save_name_arg: str = "udp_to_tcp_proxy.json"):
        """
        Constructor for UDP_2_TCP_Proxy class

        :param tcp_conn: This is the connection used for sending the TCP packets to. It is associated with tcp_sock, since tcp_conn is meant to be made with a call to socket.accept()
        :param tcp_sock: This is the socket that the proxy will include within itself. It might be made externally by some other program. If not created externally, the class will internally make the socket by itself
        :param udp_sock: This is the UDP socket that the proxy is meant to receive the UDP packets from.
        :param tcp_connection_remote_address_arg: This is the TCP address that the local TCP socket is meant to send the TCP packets to.
        :param udp_socket_remote_address_arg: This address is purely meant for debugging purposes. It does not serve any other function.
        :param udp_socket_timeout_arg: This is the timeout value that the UDP socket will have.
        :param tcp_socket_timeout_arg: This is the timeout value that the TCP socket will have.
        :param client_number_arg: This is the number assigned to the client by external entities. In real scenarios, this will be provided by the training scripts.
        :param udp_socket_local_address_arg: This is the address where the JSON files will be received from, or the UDP strings will be received in bytearray form.
        :param receiving_json_arg: This parameter denotes whether the UDP socket expects a file in JSON format, other types of files or packets. Turn this argument to True if the UDP socket should expect a JSON file. (JSON files are received via reliable UDP)
        :param device_number_arg: This argument denotes the device number used in transmission. The same device might be running more than multiple proxies or clients.
        """
        self.tcp_socket = tcp_sock  # TCP socket to send packets from
        self.tcp_connection = tcp_conn  # TCP connection to receive packets from, associated with the socket
        self.udp_socket = udp_sock  # UDP socket to receive packets from
        self.tcp_connection_remote_address = tcp_connection_remote_address_arg  # TCP address to send received UDP packets to
        self.udp_socket_remote_address = udp_socket_remote_address_arg
        self.received_string = ""
        self.received_string_buffer = []
        self.udp_socket_max_byte_size = 1800
        self.tcp_connection_local_address = tcp_connaction_local_address_arg
        self.client_number = client_number_arg
        self.udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        self.receiving_json_as_file = receiving_json_as_file_arg
        self.udp_socket_local_address = udp_socket_local_address_arg
        self.window_size_udp_socket = 20
        self.current_device = device_number_arg
        self.tcp_socket_expects_str = tcp_socket_expects_str_arg
        self.receiving_json_as_string = receiving_json_as_string_arg
        self.received_json_str = str()
        self.received_model = dict()
        self.json_file_save_name = json_file_save_name_arg
        self.json_file_handle = open(self.json_file_save_name, "w+")
        self.json_file_handle.truncate(0)
        self.pktbuf = bytearray()
        self.message = bytes()
        self.recvd_strings = []

    def init_connection_establishment(self):
        """
        This function is used to initialize connections and sockets.

        :return: None
        """
        while True:
            try:
                self.tcp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self.tcp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
                self.tcp_connection = socket.create_connection(self.tcp_connection_remote_address)
                self.udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self.udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
                print("Init connection establishment complete")
                break
            except socket.timeout:
                continue
            except ConnectionRefusedError:
                print("Endpoint is not listening for connections. Attempting to reestablish connection")
                continue

    def change_tcp_connection(self, new_tcp_connection: socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM), new_tcp_socket: socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)):
        """
        Function to change towards a new TCP connection within the proxy.

        :param new_tcp_connection: Connection object to change towards. Associated with the parameter new_tcp_socket, since the new TCP connection is made with a call to socket.accept(), after calls to socket.bind() and socket.listen() respectively.
        :param new_tcp_socket: TCP socket object to change towards.
        :return: None
        """
        self.tcp_connection = new_tcp_connection
        self.tcp_socket = new_tcp_socket
        self.tcp_connection_remote_address = self.tcp_socket.getpeername()

    def smart_reconnection(self):
        """
        Used for reconnecting to the same TCP address that was either provided within the constructor (the initialization of the object) or within a call to the function "change_tcp_connection"

        :return: None
        """
        while True:
            try:
                if self.received_string != "":
                    self.tcp_socket.sendall((self.received_string + "\n").encode())
                else:
                    self.tcp_socket.sendall("\n".encode())
                break
            except ConnectionResetError:
                self.tcp_socket.sendall("Something happened")
            except ConnectionRefusedError:
                while True:
                    try:
                        self.tcp_socket.connect(self.tcp_connection_remote_address)
                        print("Connection reestablished")
                        break
                    except ConnectionRefusedError:
                        # print("Connection refused error occurred")
                        continue
            except BrokenPipeError:
                self.tcp_socket.close()
                self.tcp_socket = socket.socket()
                self.tcp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self.tcp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
                tcp_port = 3001  # Might change due to needs
                while True:
                    try:
                        self.tcp_socket.connect(self.tcp_connection_remote_address)
                        print("Connection reestablished")
                        break
                    except ConnectionRefusedError:
                        # print("Connection refused error occurred")
                        continue
            except socket.error as err:
                print(err)
                continue

    def rcv_udp_send_tcp(self):
        """
        This is the driving function for the entire class. If you are planning to somehow thread this class, use this function as the target function.

        :return: None
        """
        print("Attempting to establish connection to ", self.tcp_connection_remote_address, "... please wait.")
        self.smart_reconnection()
        print("Connection established via socket object ", self.tcp_socket)
        print("TCP bindings finished")

        if not self.receiving_json_as_file:
            while True:
                try:
                    self.udp_socket.bind(self.udp_socket_local_address)
                    print("Bound to UDP socket")
                    break
                except ConnectionRefusedError:
                    print("Connection refused error at UDP socket")
                    continue
                except KeyboardInterrupt:
                    self.tcp_socket.close()
                    self.udp_socket.close()
                    sys.exit(0)

        print("UDP packets expected from ", self.udp_socket_remote_address)

        while True:
            try:
                if not self.receiving_json_as_file:
                    self.received_string = self.udp_socket.recv(self.udp_socket_max_byte_size).decode()
                    if self.received_string == "" or self.received_string == "\n":
                        continue
                    elif self.received_string == "break" or self.received_string == "break\n":
                        if self.receiving_json_as_string:
                            self.received_json_str = "".join(self.received_string_buffer)
                            self.received_model = json.loads(self.received_json_str)
                            json.dump(self.received_model, self.json_file_handle)
                        self.received_string = "".join(self.received_string_buffer)
                        print("Received all the supposed packets.")
                        break
                    else:
                        try:
                            self.received_string_buffer.append(self.received_string)
                            self.tcp_socket.sendto(self.received_string.encode(), self.tcp_connection_remote_address)
                        except BrokenPipeError:
                            self.smart_reconnection()
                        print(self.received_string)
                        continue
                else:
                    self.received_string = self.udp_socket.recv(self.udp_socket_max_byte_size).decode()

            except socket.timeout:
                now = datetime.now()
                print("Nothing received at ", now)
                continue
            except KeyboardInterrupt:
                self.udp_socket.close()
                self.tcp_socket.close()
                sys.exit(0)

    def recv_any_bytes_and_send_it(self):
        self.udp_socket.bind(self.udp_socket_local_address)
        print("UDP packets are received from address: ", self.udp_socket_local_address, "TCP packets are sent to address ", self.tcp_connection_remote_address)
        while True:
            try:
                recvd_message, recvd_from = self.udp_socket.recvfrom(65536)
                print("Received something")
                try:
                    decoded_msg = recvd_message.decode()
                    if "done" in decoded_msg:
                        self.tcp_connection.sendto(self.pktbuf, self.tcp_connection_remote_address)
                        print("Sent all the packets in pktbuf")
                        self.pktbuf = bytearray()
                        continue
                    else:
                        pass
                except:
                    pass
                self.pktbuf.extend(recvd_message)
                print("Received something into the pktbuf")
            except socket.timeout:
                continue
            except KeyboardInterrupt:
                sys.exit(0)
            except:
                print(sys.exc_info())
                continue

    def recv_pickled_data_send_udp_data(self):
        self.message = communication.receive_data(self.tcp_connection)

    def send_single_packet_to_router(self, curr_packet: str = "", pkt_len: int = 1500):
        print("Test source to MK1 on %s:%d" % (self.udp_socket_remote_address[0], self.udp_socket_remote_address[1]))
        Target = self.udp_socket_remote_address
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

    def recv_udp_data_send_tcp_data(self, mk5_used=True):
        self.udp_socket.bind(self.udp_socket_local_address)
        print("Receiving from UDP socket address: ", self.udp_socket_local_address, " sending to TCP socket address: ", self.tcp_connection_remote_address)
        while True:
            recvd_message, recvd_from = self.udp_socket.recvfrom(65536)
            if mk5_used:
                recvd_message_decoded = keep_useful_characters(strip_away_indicators(input_string=chr_long_string(bytearray(recvd_message)), indicator_string="^||"))
                print("Received ", recvd_message_decoded)
                if len(recvd_message_decoded) < 50 and "0" not in recvd_message_decoded and "1" not in recvd_message_decoded:
                    print("Got wrong message")
                if "DONE" in recvd_message_decoded:
                    communication.send_data(self.tcp_connection, bytes("".join(self.recvd_strings), "utf-8"))
                    self.recvd_strings = []
                    continue
                else:
                    self.recvd_strings.append(recvd_message_decoded)
                    continue
                pass
            else:

                pass
        pass


if __name__ == "__main__":
    test_object = UDP_2_TCP_Proxy(tcp_connection_remote_address_arg=("127.0.0.1", 10001), udp_socket_remote_address_arg=("192.168.1.31", 4000), udp_socket_local_address_arg=("192.168.1.57", 4001))
    test_object.init_connection_establishment()
    test_object.recv_udp_data_send_tcp_data()
    pass
