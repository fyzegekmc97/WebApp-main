import sys
import datetime


def send_data(connection, data):
    # print(len(data))
    data_length = len(data).to_bytes(4, byteorder='big')
    while True:
        try:
            connection.sendall(data_length + data)
            now = datetime.datetime.now()
            now_str = now.strftime("%d/%m/%Y %H:%M:%S")
            print("Data sent at ", now_str, " of length ", len(data))
            break
        except KeyboardInterrupt:
            sys.exit(0)
        except BrokenPipeError:
            continue
    # connection.sendall(data_length + str.encode(data))


"""
def send_data_to_all(connections, data):
    # print(len(data))
    data_length = len(data).to_bytes(4, byteorder='big')
    for connection in connections:
        connection.sendall(data_length + str.encode(data))
"""


def receive_data(connection):
    data_length_raw = bytes()
    while True:
        try:
            data_length_raw = connection.recv(4)
            if data_length_raw:
                print("Expecting to receive ", int.from_bytes(data_length_raw, "big"), " bytes of data")
            break
        except BrokenPipeError:
            print("Other endpoint has broken the connection.")
            continue
        except KeyboardInterrupt:
            print("Keyboard interrupt raised")
            sys.exit(0)

    if data_length_raw is None or not data_length_raw:
        return None
    data_length = int.from_bytes(data_length_raw, "big")
    data = bytearray()
    while len(data) < data_length:
        packet = connection.recv(data_length - len(data))
        if not packet:
            return None
        data.extend(packet)
    return data
