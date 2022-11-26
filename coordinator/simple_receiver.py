from imp import reload
import socket
import time
import tqdm
import zipfile
import os
from keras import models, losses, metrics, layers
# device's IP address
local_ip = "127.0.0.1"
local_port = 5001
buffer_size = 65536
s = socket.socket()
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind((local_ip, local_port))
s.listen(5)
print(f"[*] Listening as {local_ip}:{local_port}")
client_socket, address = s.accept()
print(f"[+] {address} is connected.")
with open("received_model.zip", "wb") as f:
    while True:
        # read 1024 bytes from the socket (receive)
        bytes_read = client_socket.recv(buffer_size)
        print("Received something", len(bytes_read))
        f.write(bytes_read)
        try:
            if len(bytes_read) == 0:
                print("Done")
                break
            else:
                continue
        except UnicodeDecodeError:
            continue
f.close()
s.close()
client_socket.close()

with zipfile.ZipFile("received_model.zip", 'r') as zip_ref:
    zip_ref.extractall("received_model")
zip_ref.close()
json_file = open("received_model/model.json", "r")
model_json = json_file.read()
json_file.close()
loaded_model = models.model_from_json(model_json)
loaded_model.load_weights("received_model/model.h5")
print(loaded_model.weights)
print(len(loaded_model.weights))
print(len(loaded_model.layers))
model_dict = {"weights":[]}
# for i in range(len(loaded_model.weights)):
#    model_dict["weights"].append(loaded_model.weights)
# close the client socket