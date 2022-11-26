import json
import os
import socket
import sys
import textwrap
import threading
import numpy as np
import random
import time
import numpy.random as npyrnd
from numpy import dtype as npydtype
from worker_udp_sender import Smart_UDP_Sender_Socket
import paramiko
import select
from worker_udp_receiver import AlwaysOpenUDP_Receiver
import numpy.ma as ma
import zlib
from math import log2, ceil
from typing import Tuple
import rsa


class ReceivingSocketThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.some_threaded_socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
        self.some_threaded_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.some_threaded_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        self.some_threaded_socket.bind(("192.168.1.57", 4001))
        self.some_threaded_socket.setblocking(True)
        self.some_threaded_socket.settimeout(1.0)
        self.packets = []

    def run(self) -> None:
        try:
            successful_packets = 0
            while successful_packets < 100:
                try:
                    recvd_bytes, recvd_from = self.some_threaded_socket.recvfrom(65536)
                    successful_packets += 1
                    self.packets.append(recvd_bytes)
                    print("Received")
                except socket.timeout:
                    continue
                except KeyboardInterrupt:
                    print(self.packets)
                    sys.exit(0)
        except KeyboardInterrupt:
            print(self.packets)
            sys.exit(0)


class SendingSocketThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self) -> None:
        successful_packets = 0
        while successful_packets < 100:
            try:
                send_test_string()
                successful_packets += 1
                print("Sent")
            except:
                continue


test_string = "abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz"


def send_test_string():
    sender = Smart_UDP_Sender_Socket(udp_local_address_arg=("192.168.1.57", 4002),
                                     udp_target_address_arg=("192.168.1.41", 4000))
    sender.bind_to_local_address()
    sender.send_single_packet_to_router(curr_packet=test_string)


def receive_test_string():
    some_socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
    some_socket.settimeout(100.0)
    some_socket.bind(("192.168.1.57", 4001))
    recvd_bytes, recvd_from = some_socket.recvfrom(1500)
    print(recvd_bytes)
    print(recvd_from)


def send_long_string_tcp():
    test_string = ""
    for i in range(125536):
        test_string += "a"
    some_socket = socket.socket()
    some_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    some_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    some_socket.connect(("127.0.0.1", 5000))
    while True:
        try:
            time.sleep(1.0)
            packets = textwrap.wrap(test_string, 2000)
            for i in range(len(packets)):
                str_to_send = packets[i] + "mystr"
                some_socket.send(str_to_send.encode())
                print("Transmitted a string of length ", len(str_to_send))
            break
        except:
            print("Exception raised")
            sys.exit(0)


def send_single_packet_reliably(packet: bytes, local_address: tuple = ("172.23.176.61", 9000), remote_address: tuple = ("172.23.176.60", 9000)):
    some_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    some_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    some_socket.settimeout(1.0)
    some_socket.bind(local_address)
    print("Socket bound")
    global recvd_bytes
    while True:
        try:
            some_socket.sendto(packet,remote_address)
            print("Packet sent")
            recvd_bytes = some_socket.recv(1500)
            break
        except:
            continue
    print(recvd_bytes.decode())
    some_socket.close()


def send_file_length(some_socket: socket.socket, data_length: int, remote_address: Tuple[str,int]):
    some_socket.sendto(("file_len_" + str(data_length)).encode(), remote_address)


def send_file(some_socket: socket.socket, data_length: int, remote_address: Tuple[str,int], data: bytes, packet_size: int = 1500):
    total_packet_count = ceil(data_length / packet_size)
    for i in range(0,total_packet_count):
        if i != total_packet_count - 1:
            some_socket.sendto(data[packet_size*i:i*packet_size+packet_size], remote_address)
        else:
            some_socket.sendto(data[packet_size*i:], remote_address)
        time.sleep(0.5)


def send_ID(some_socket: socket.socket, ID_number: int, remote_address: Tuple[str,int], ID_keyword: str):
    some_socket.sendto((str(ID_number) + ID_keyword).encode(), remote_address)


if __name__ == "__main__":
    pass
    
    

