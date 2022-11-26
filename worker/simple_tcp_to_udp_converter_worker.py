import socket
import sys

listening_tcp_socket = socket.socket()
listening_tcp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
listening_tcp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
sending_udp_socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
sending_udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sending_udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)

tcp_listening_socket_local_ip_address = "127.0.0.1"
tcp_listening_socket_local_port = 5000

udp_sending_socket_local_ip_address = "127.0.0.1"
udp_sending_socket_local_port = 6250

udp_sending_socket_remote_ip_address = "127.0.0.1"
udp_sending_socket_remote_port = 6500

while True:
    try:
        sending_udp_socket.bind((udp_sending_socket_local_ip_address, udp_sending_socket_local_port))
        break
    except:
        continue

while True:
    try:
        listening_tcp_socket.bind((tcp_listening_socket_local_ip_address, tcp_listening_socket_local_port))
        break
    except:
        continue

while True:
    try:
        listening_tcp_socket.listen()
        break
    except:
        continue

global obtained_listening_connection_tcp_socket

while True:
    try:
        print("Possibly a blocking operation")
        obtained_listening_connection_tcp_socket, remote_address = listening_tcp_socket.accept()
        break
    except KeyboardInterrupt:
        break
    except:
        continue

while True:
    try:
        my_bytez = obtained_listening_connection_tcp_socket.recv(65536)
        sending_udp_socket.sendto(my_bytez, (udp_sending_socket_remote_ip_address, udp_sending_socket_remote_port))
    except KeyboardInterrupt:
        break
    except:
        continue


