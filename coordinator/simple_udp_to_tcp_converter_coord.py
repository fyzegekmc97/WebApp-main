import socket
import sys

listening_udp_socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
listening_udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
listening_udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
sending_tcp_socket = socket.socket()
sending_tcp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sending_tcp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)

tcp_sending_socket_local_ip_address = "127.0.0.1"
tcp_sending_socket_local_port = 5250

udp_listening_socket_local_ip_address = "127.0.0.1"
udp_listening_socket_local_port = 6500

tcp_sending_socket_remote_ip_address = "127.0.0.1"
tcp_sending_socket_remote_port = 6000

while True:
    try:
        sending_tcp_socket.bind((tcp_sending_socket_local_ip_address, tcp_sending_socket_local_port))
        break
    except:
        print(sys.exc_info())
        continue


while True:
    try:
        listening_udp_socket.bind((udp_listening_socket_local_ip_address, udp_listening_socket_local_port))
        break
    except:
        print(sys.exc_info())
        continue

while True:
    try:
        sending_tcp_socket.connect((tcp_sending_socket_remote_ip_address, tcp_sending_socket_remote_port))
        break
    except:
        print(sys.exc_info())
        continue

while True:
    try:
        bytez = listening_udp_socket.recv(65536)
        sending_tcp_socket.sendto(bytez, (tcp_sending_socket_remote_ip_address, tcp_sending_socket_remote_port))
        continue
    except KeyboardInterrupt:
        break
    except:
        continue
