import json
import os
import pickle
import signal
import socket
import time
from typing import *
from threading import *
import sys
import textwrap
import logging


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


class Smart_UDP_Sender_Socket:
    def __init__(self, udp_socket_arg: socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM) = None, udp_local_address_arg: Tuple[str, int] = ("127.0.0.1", 5200), udp_target_address_arg: Tuple[str, int] = ("127.0.0.1", 5250), should_send_pickled_data_arg: bool = False, testing_mode_on_arg: bool = True, test_string_arg: str = str(0), sending_json_arg: bool = True, json_file_location_arg: str = "unpickled_message_json_device0client0.json", intra_packet_time_arg: float = 0.05, grace_period_arg: float = 3.0, socket_timeout_arg: float = 5.0, client_number_arg: int = 0, sending_to_router_arg: bool = True):
        """
        A class written to use for testing purposes, although the class name includes 'Smart', UDP isn't that smart and there is not much to do if the remote side disconnects, only precautions can be taken to reestablish the connection to the local socket (or address).

        :param udp_socket_arg: This argument is used only when the socket is created externally by some other entity. If no argument is provided, the class will create a socket on its own.
        :param udp_target_address_arg: This argument is used to denote the remote device's address.
        :param should_send_pickled_data_arg: This argument is used to denote whether the socket should send the data in pickled form.
        :param testing_mode_on_arg: This argument denotes whether the socket is being used for testing purposes.
        :param test_string_arg: This argument is used to relay or hand out a sample string for testing purposes. Can be left empty
        :param sending_json_arg: This argument denotes whether the smart UDP packet sending socket is supposed to send JSON files or JSON strings.
        :param json_file_location_arg: This argument denotes the location of the JSON file that would be sent if he socket is meant to send a JSON file or string. The argument "sending_json_arg" must be set to true in the constructor or the property "sending_json" should be manually set to true by the calling entity.
        :param intra_packet_time_arg: This is the time taken between each sent UDP packet.
        :param grace_period_arg: This is the amount of time waited after a call to send packets is made. After a call to send packets is made, the amount of time set by this argument is waited before sending the packets, to give time to the receiver. The grace period of the socket could also be set manually by its property "grace_period"
        """
        if udp_socket_arg is not None:
            self.udp_socket = udp_socket_arg
        else:
            self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp_target_address = udp_target_address_arg
        self.udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        self.should_run = True
        self.execution_stopping_reason = ""
        self.should_send_pickled_data = should_send_pickled_data_arg
        self.testing_mode_on = testing_mode_on_arg
        self.client_number = client_number_arg
        self.test_string = test_string_arg
        self.pickled_test_string = bytes()
        self.sending_json = sending_json_arg
        self.json_file_location = json_file_location_arg
        self.udp_socket_local_address = udp_local_address_arg
        temp_file_handle = open(self.json_file_location, "a+")
        temp_file_handle.truncate(0)
        json_obj = {"weights": [], "biases": []}
        json.dump(json_obj, temp_file_handle)
        temp_file_handle.close()
        self.json_file_handle = open(self.json_file_location, "r")
        self.json_data_in_file = json.load(self.json_file_handle)
        self.json_file_handle.close()
        self.json_file_handle = open(self.json_file_location, "a+")
        self.json_file_handle.truncate(0)
        self.json_file_handle.close()
        self.max_udp_packet_size = 1500
        self.intra_packet_time = intra_packet_time_arg
        self.grace_period = grace_period_arg
        self.acked_packets_index_list = []
        self.reliably_sent_packet_count = 0
        self.ack_sent_or_not = False
        self.sending_to_router = sending_to_router_arg
        self.handshake_attempt_count = 0

    def send_single_packet_to_router(self, curr_packet: str = "", pkt_len: int = 1600):
        print("Test source to MK1 on %s:%d" % (self.udp_target_address[0], self.udp_target_address[1]))
        pkt_count = 0
        # Open the socket to communicate with the mk1
        Target = (self.udp_target_address[0], self.udp_target_address[1])  # the target address and port
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
            return pkt_count
        except KeyboardInterrupt:
            self.udp_socket.close()
            print('User CTRL-C, stopping packet source')
            sys.exit(1)
        except:
            self.udp_socket.close()
            print("Got exception:", sys.exc_info()[0])
            raise

    def receive_OK_from_router(self):
        try:
            recvd_ok, recvd_from = self.udp_socket.recvfrom(65536)
            recvd_ack_decoded = chr_long_string(bytearray(recvd_ok))
            print(recvd_ack_decoded)
            if "OK" in recvd_ack_decoded:
                print(recvd_ack_decoded)
                return True
            else:
                print("Got something else than OK.")
                return False
        except KeyboardInterrupt:
            sys.exit(0)
        except socket.timeout:
            print("Did not get OK, socket timed out.", self.udp_socket)

    def send_ACK_to_router(self, ack_index_arg: Optional[int] = None):
        ACK_index = 0
        if ack_index_arg is not None:
            ACK_index = ack_index_arg
        ack_sent = self.send_single_packet_to_router(curr_packet=("ACK " + str(ACK_index) + " "))
        self.ack_sent_or_not = bool(ack_sent)

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
        print("Test source to MK1 on %s:%d" % (self.udp_target_address[0], self.udp_target_address[1]))
        self.udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        Target = self.udp_target_address
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

    def receive_ACK_from_router(self):
        try:
            recvd_ack, recvd_from = self.udp_socket.recvfrom(65536)
            recvd_ack_decoded = chr_long_string(bytearray(recvd_ack))
            if "ACK" in recvd_ack_decoded:
                print(recvd_ack_decoded)
                return True
            else:
                print("Got something else than ACK: ", recvd_ack_decoded)
                return False
        except KeyboardInterrupt:
            sys.exit(0)
        except socket.timeout:
            print("Did not get ACK, socket timed out.")

    def initiate_handshake(self):
        while True:
            try:
                if self.sending_to_router:
                    print("Coordinator attempting handshake with router...")
                    self.udp_socket.settimeout(2.0)
                    print("Coordinator sender socket configuration complete")
                    self.handshake_attempt_count += 1
                    self.send_single_packet_to_router(curr_packet="Coordinator initiated handshake", pkt_len=1500)
                    print("Coordinator sender sent its packet to router.")
                    break
                else:
                    message, recvd_from = self.udp_socket.recvfrom(65536)
                    self.udp_socket.sendto("Coordinator initiated handshake".encode(), self.udp_target_address)
                    if message.decode() == "OK" or "OK" in message.decode():
                        print("OK message received into the coordinator sender from: ", recvd_from)
                        if self.udp_target_address == recvd_from:
                            print("Remote address was as expected, not changing remote address...")
                            pass
                        else:
                            print("Changing target address to: ", recvd_from)
                            self.udp_target_address = recvd_from
                        break
                    else:
                        print("Got wrong message into the coordinator sender, message was: ", message.decode())
                        continue

            except socket.timeout:
                if self.handshake_attempt_count > 10:
                    break
                continue

    def send_multiples_of_a_packet(self, curr_packet_arg: str = "", repeat_count: int = 3):
        for i in range(repeat_count):
            self.send_single_packet_to_router(curr_packet=curr_packet_arg)

    def bind_to_local_address(self):
        try:
            self.udp_socket.bind(self.udp_socket_local_address)
            print("Parameter server sender socket for client ID ", self.client_number, " will receive ACK's, NACK's and send packets from: ", self.udp_socket_local_address)
        except OSError:
            for i in range(1025, 65536):
                try:
                    self.udp_socket.bind((self.udp_socket_local_address[0], i))
                    self.udp_socket_local_address = (self.udp_socket_local_address[0], i)
                    print("Parameter server sender socket for client ID ", self.client_number, " will receive ACK's, NACK's and send packets from: ", self.udp_socket_local_address)
                except OSError:
                    continue

    def start_sending(self, new_json_file_location: Optional[str] = None, dict_to_send: dict = None):
        print("Sending has started from the coordinator...")
        if new_json_file_location is not None and os.path.exists(new_json_file_location):
            self.json_file_location = new_json_file_location
            self.json_file_handle = open(self.json_file_location, "r")
            print("JSON file location is: ", self.json_file_location)
            some_json_str = self.json_file_handle.read()
            self.json_data_in_file = json.dumps(some_json_str)
            self.json_file_handle.close()
            self.json_file_handle = open(self.json_file_location, "a+")
            self.json_file_handle.truncate(0)
            self.json_file_handle.close()
        elif new_json_file_location is not None and not os.path.exists(new_json_file_location):
            self.json_file_location = new_json_file_location
            temp_file_handle = open(self.json_file_location, "a+")
            temp_file_handle.close()
            self.json_file_handle = open(self.json_file_location, "r")
            self.json_data_in_file = json.load(self.json_file_handle)
            self.json_file_handle.truncate(0)
            self.json_file_handle.close()
        else:
            print("File does not exist, keeping old location: ", self.json_file_location)

        if dict_to_send is None:
            pass
        else:
            self.json_data_in_file = str(dict_to_send)

        while self.should_run:
            try:
                if self.should_send_pickled_data:
                    if self.testing_mode_on:
                        """
                        For testing purposes, this class should continuously send the client number. The model parameters are already sent via reliable UDP anyways. The proxy at the worker side does not need to listen to the port for the client number in real-life applications all the time anyways, unlike in the testing scenario. In a testing scenario, this script only sends the client number to the proxy code.
                        """
                        print("Sent pickled data to address ", self.udp_target_address)
                        self.pickled_test_string = pickle.dumps(self.test_string)
                        self.udp_socket.sendto(self.pickled_test_string, self.udp_target_address)
                        time.sleep(1)
                        continue
                    else:
                        print("This class is not implemented for non-testing purposes as of now")
                        break
                else:
                    if self.sending_json:
                        if self.sending_to_router:
                            pass
                        else:
                            print("Parameter server sender socket sending packets to: ", self.udp_target_address, "from address: ", self.udp_socket.getsockname())
                            print("Make sure the receiver is run first. Giving grace period of", str(self.grace_period), "seconds.")
                            time.sleep(self.grace_period)
                            udp_packet_list_to_send = textwrap.wrap(str(self.json_data_in_file), self.max_udp_packet_size)
                            sig_decoded = ""
                            try:
                                send_signal, recvd_from = self.udp_socket.recvfrom(65536)
                                self.udp_target_address = recvd_from
                                sig_decoded = send_signal.decode()
                                print("Received UDP packet from address: ", recvd_from, " listen to that address for upcoming packets. Packet content was: ", sig_decoded)
                            except socket.timeout:
                                self.udp_socket.sendto("Ready signal not received.".encode(), self.udp_target_address)
                                while True:
                                    try:
                                        send_signal, recvd_from = self.udp_socket.recvfrom(65536)
                                        self.udp_target_address = recvd_from
                                        sig_decoded = send_signal.decode()
                                        print("Received UDP packet from address: ", recvd_from, " listen to that address for upcoming packets. Packet content was: ", sig_decoded)
                                        break
                                    except socket.timeout:
                                        continue

                                if sig_decoded == "ACK" or sig_decoded == "ACK 0" or sig_decoded == "ready" or "ready" in sig_decoded:
                                    pass
                                else:
                                    while True:
                                        time.sleep(self.intra_packet_time)
                                        try:
                                            send_signal, recvd_from = self.udp_socket.recvfrom(65536)
                                        except socket.timeout:
                                            continue
                                        sig_decoded = send_signal.decode()
                                        if sig_decoded == "ACK" or sig_decoded == "ACK 0" or sig_decoded == "ready" or "ready" in sig_decoded:
                                            break

                            for i in range(len(udp_packet_list_to_send)):
                                self.udp_socket.sendto(udp_packet_list_to_send[i].replace("'", "\"").encode(), self.udp_target_address)
                                print("Sent ", len(udp_packet_list_to_send[i]), " bytes of data to ", self.udp_target_address, " session packet with index ", i)
                                ack_signal = bytes()
                                try:
                                    ack_signal, recvd_from = self.udp_socket.recvfrom(65536)
                                except socket.timeout:
                                    print(self.udp_socket.timeout)
                                    print("ACK not received to the coordinator")
                                ack_signal_decoded = ack_signal.decode()
                                if "ACK" in ack_signal_decoded:
                                    print("ACK received. Amount of packets sent reliably thus far: ", self.reliably_sent_packet_count + 1, ". ACK index was: ", ack_signal_decoded.split(" ")[1])
                                    self.acked_packets_index_list.append(ack_signal_decoded.split(" ")[1])
                                    self.reliably_sent_packet_count += 1
                                    pass
                                else:
                                    print("Sent " + str(self.reliably_sent_packet_count) + " packets reliably before reaching an issue.")
                                    while True:
                                        try:
                                            # Write the code for not receiving ACK for a certain message
                                            break
                                        except socket.timeout:
                                            # Handle another unreceived ACK here.
                                            pass
                                    pass
                                time.sleep(self.intra_packet_time)
                            for i in range(4):
                                self.udp_socket.sendto("break".encode(), self.udp_target_address)
                            break
                    else:
                        self.test_string = "Sending non-pickled data"
                        print("Sent to address ", self.udp_target_address)
                        self.udp_socket.sendto(self.test_string.encode(), self.udp_target_address)
                        time.sleep(self.intra_packet_time)
                        continue
            except KeyboardInterrupt:
                self.udp_socket.close()
                self.execution_stopping_reason = "Keyboard interrupt"
                os.kill(os.getpid(), signal.SIGKILL)
                break
            except ConnectionResetError:
                self.udp_socket.close()
                self.execution_stopping_reason = "Connection reset"
                break
        print(self.execution_stopping_reason)

    def send_exit_signal(self):
        for i in range(4):
            self.udp_socket.sendto("exit".encode(), self.udp_target_address)

    def close_socket(self):
        self.udp_socket.close()


class AlwaysSendingUDP_Sender(Thread):
    def __init__(self, local_ip_address: str = "192.168.1.57", remote_ip_address: str = "192.168.1.31", local_port: int = 4002, remote_port: int = 4000):
        Thread.__init__(self)
        self.local_ip = local_ip_address
        self.local_port = local_port
        self.remote_ip = remote_ip_address
        self.remote_port = remote_port
        self.udp_socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
        self.udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        self.udp_socket.settimeout(1.0)
        self.packets_to_send = []
        self.dict_to_send = dict()
        self.max_packet_length = 1500
        self.idle = True
        self.sending = False
        self.done = False

    def bind_to_local_ip_address(self):
        self.udp_socket.bind((self.local_ip, self.local_port))

    def set_dictionary_to_send(self, dict_to_send_arg: dict):
        self.dict_to_send = dict_to_send_arg

    def send_single_packet_to_router(self, curr_packet: str = "", pkt_len: int = 1600):
        print("Test source to MK1 on %s:%d" % (self.remote_ip, self.remote_port))
        pkt_count = 0
        # Open the socket to communicate with the mk1
        Target = (self.remote_ip, self.remote_port)  # the target address and port
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
            return pkt_count
        except KeyboardInterrupt:
            self.udp_socket.close()
            print('User CTRL-C, stopping packet source')
            sys.exit(1)
        except:
            print("Got exception:", sys.exc_info()[0])
            raise

    def run(self) -> None:
        print("Sending started...")
        index = 0
        while True:
            if len(self.packets_to_send) == 0:
                if len(self.dict_to_send) == 0:
                    print("No dictionary provided")
                    return
                else:
                    self.packets_to_send = textwrap.wrap(str(self.dict_to_send), self.max_packet_length)
                    print("Packets to send are ready")
            if self.idle:
                try:
                    recvd_message, recvd_from = self.udp_socket.recvfrom(65536)
                    self.idle = False
                    self.sending = True
                    print("Received message from ", recvd_from, " exiting idle mode. Message received was of length ", len(recvd_message))
                except:
                    index = 0
                    print("Did not receive ready message, still in idle mode")
                    continue
            elif self.sending:
                try:
                    self.send_single_packet_to_router(curr_packet=self.packets_to_send[index])
                    print("Sent packet of length ", len(self.packets_to_send[index]), "to address", (self.local_ip, self.local_port))
                    recvd_message, recvd_from = self.udp_socket.recvfrom(65536)
                    print("ACK received from ", recvd_from)
                    index += 1
                    if index == len(self.packets_to_send):
                        self.sending = False
                        self.idle = True
                        self.done = True
                    else:
                        self.sending = True
                        self.idle = False
                        self.done = False
                except socket.timeout:
                    self.sending = True
                    self.idle = False
                    self.done = False
                    continue
                except KeyboardInterrupt:
                    self.udp_socket.close()




if __name__ == "__main__":
    sender_thread_obj = UDP_Sender_Thread()
    sender_thread_obj.start()
    try:
        sender_thread_obj.join()
    except SystemExit:
        print("System exit raised")
        sys.exit(0)
    except SystemError:
        print("System error raised")
        sys.exit(0)
    except KeyboardInterrupt:
        sender_thread_obj.sender.should_run = False
        time.sleep(1)
        print("Keyboard interrupt raised")
        sender_thread_obj.sender.execution_stopping_reason = "Keyboard interrupt"
        os.kill(os.getpid(), signal.SIGKILL)
        sys.exit(0)
