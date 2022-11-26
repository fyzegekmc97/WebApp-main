import math
import os
import socket
import textwrap
import time
from typing import Tuple, Optional, Any
import sys
from typing import List
import communication
import pickle
from past.builtins import raw_input


# enable soft-wrapping in an editor please
def download_model_json():
    os.system("python2.7 server_reliableUDP.py")


class TCP_2_UDP_Proxy:
    """
    This is an object/class solely written/created for encapsulating a TCP packet into a UDP packet. It is open to improvement/development. Refer to the documentation for the constructor for creating an object of this type.
    """
    def __init__(self, tcp_conn: socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM) = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM), tcp_sock: socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM) = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM), udp_sock: socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM) = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM), tcp_socket_local_address_arg: Tuple[str, int] = ("127.0.0.1", 4001), tcp_socket_remote_address_arg: Tuple[str, int] = ("127.0.0.1", 5050), udp_socket_remote_address_arg: Tuple[str, int] = ("127.0.0.1", 5100), udp_socket_local_address_arg: Tuple[str, int] = ("127.0.0.1", 1500), tcp_socket_received_strings_buffer_arg: List[str] = None, tcp_socket_timeout_arg: float = 10.0, udp_socket_timeout_arg: float = 10.0, receiving_tf_variables_arg: Optional[bool] = None, sending_model_as_json_arg: Optional[bool] = None):
        """
            :parameter tcp_conn: This is the TCP connection object that the proxy will include within itself. It should be indirectly associated with the parameter "tcp_sock", via a call to either socket.create_connection() or socket.accept(). If not created externally, the proxy will create its own TCP socket and then connection for TCP packet receival.
            :param tcp_sock: This is the TCP socket object that the proxy will include within itself. If not created externally, the proxy will automatically create its own TCP socket.
            :param udp_sock: This is the UDP socket object that will be responsible for sending the UDP-encapsulated packets, after receiving the TCP packets. If not created externally, it will be created internally.
            :param tcp_socket_local_address_arg: This is the address that the TCP socket will listen from. External connections should be made targeting this address. External devices/objects should "connect to" this address.
            :param tcp_socket_remote_address_arg: This is the TCP address that the TCP socket would get its TCP connection and packets from. This argument is currently used purely for debugging reasons, in the future it might be used for other purposes.
            :param udp_socket_remote_address_arg: This is the address that the UDP socket will send its encapsulated packets to.
            :param
            """
        self.packets = []
        self.max_udp_packet_size = 1500
        self.decoded_message_str = ""
        self.decoded_message = bytes()
        self.message = bytes()
        self.tcp_connection = tcp_conn
        self.tcp_socket = tcp_sock
        self.udp_socket = udp_sock
        self.tcp_socket_local_address = tcp_socket_local_address_arg
        self.tcp_socket_remote_address = tcp_socket_remote_address_arg
        self.configure_connections_and_sockets()
        self.udp_socket_remote_address = udp_socket_remote_address_arg
        self.udp_socket_local_address = udp_socket_local_address_arg
        tcp_socket_received_strings_buffer_arg = []
        self.tcp_socket_received_strings_buffer = tcp_socket_received_strings_buffer_arg
        self.received_data = ""
        self.tcp_socket_timeout = tcp_socket_timeout_arg
        self.udp_socket_timeout = udp_socket_timeout_arg
        self.receiving_tf_variables = bool()
        self.local_model = Any
        self.client_number = int()
        self.pktbuf = bytearray()
        if receiving_tf_variables_arg is not None:
            self.receiving_tf_variables = receiving_tf_variables_arg
        else:
            self.receiving_tf_variables = True

        self.sending_model_as_json = bool()
        if sending_model_as_json_arg is not None:
            self.sending_model_as_json = sending_model_as_json_arg
        else:
            self.sending_model_as_json = True

    def configure_connections_and_sockets(self):
        while True:
            try:
                self.tcp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self.tcp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
                self.tcp_socket.bind(self.tcp_socket_local_address)
                self.tcp_socket.listen(1)
                self.tcp_connection, self.tcp_socket_remote_address = self.tcp_socket.accept()
                break
            except socket.timeout:
                continue
            except OSError:
                continue


    def change_tcp_connection_external(self, new_tcp_conn: socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM), new_tcp_sock: socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)):
        self.tcp_connection = new_tcp_conn
        self.tcp_socket = new_tcp_sock

    def change_tcp_connection_internal(self, new_tcp_remote_address: Tuple[str, int]):
        self.tcp_socket_remote_address = new_tcp_remote_address
        self.tcp_connection = self.tcp_socket.connect(self.tcp_socket_remote_address)

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

    def rcv_tcp_and_send_udp(self):
        # UDP stuff and bindings
        print("UDP target IP: ", self.udp_socket_remote_address[0])
        print("UDP target port: ", self.udp_socket_remote_address[1])
        socket_problem_persists = False
        print("TCP socket got connection from address: ", self.tcp_connection.getpeername())
        udp_socket_max_size = 1800
        while True:
            try:
                if self.receiving_tf_variables:
                    # Receiving either the client ID, or some model
                    data = communication.receive_data(self.tcp_connection)
                    if data is not None and len(data) != 0:
                        self.local_model = pickle.loads(data)
                        print(self.local_model)
                        print(type(self.local_model))
                        if type(self.local_model) is int:
                            self.client_number = self.local_model
                            print("Got client number: ", self.client_number)
                        elif type(self.local_model) is dict:
                            print("Got model type.")
                            # Send model via UDP here

                    continue
                else:
                    # Receiving basic string from the TCP socket.
                    self.received_data = self.tcp_connection.recv(65536).decode()
                    if self.received_data == "" or self.received_data == "\n":
                        print("Nothing received, waiting for new packets...")
                        # self.tcp_socket.listen(1)
                        # tcp_connection, addr = self.tcp_socket.accept()
                        self.configure_connections_and_sockets()
                        time.sleep(1)
                        pass
                    else:
                        model_data_list = textwrap.wrap(self.received_data, udp_socket_max_size)
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
                continue
            except KeyboardInterrupt:
                self.tcp_connection.close()
                self.tcp_socket.close()
                self.udp_socket.close()
                sys.exit(0)

    def recv_any_byte_and_send_it(self, mk5_used: bool = True):
        print("Receiving to TCP address ", self.tcp_connection.getsockname(), " sending to UDP address ",
              self.udp_socket_remote_address)
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
                        self.send_single_packet_to_router(curr_packet=self.decoded_message_str,
                                                          pkt_len=len(self.decoded_message_str) + 100)
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
    test_object = TCP_2_UDP_Proxy(tcp_socket_local_address_arg=("127.0.0.1", 5000), tcp_socket_remote_address_arg=("127.0.0.1", 5050), udp_socket_remote_address_arg=("192.168.1.31", 4000), udp_socket_local_address_arg=("192.168.1.57", 3000))
    test_object.recv_any_byte_and_send_it()
    pass
