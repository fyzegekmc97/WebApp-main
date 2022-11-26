import sys
import socket
import joblib
import numpy as np
import tensorflow as tf
from keras import models, metrics, losses
from sys import getsizeof
import pickle
import zipfile
import os
import tqdm

seperator = "<SEPARATOR>"
buffer_size = 1500

remote_host = "127.0.0.1"
remote_port = 5001

s = socket.socket()

filename = "myzipfile.zip"

filesize = os.path.getsize(filename)

s.connect((remote_host, remote_port))
print("[+] Connected.")

my_model = models.load_model("my_model")
print(len(my_model.weights))
print(type(my_model.weights))
print(getsizeof(my_model))

progress = tqdm.tqdm(range(filesize), f"Sending {filename}", unit="B", unit_scale=True, unit_divisor=1024)
with open(filename, "rb") as f:
    while True:
        bytes_read = f.read(buffer_size)
        if not bytes_read:
            # file transmitting is done
            break
        # we use sendall to assure transimission in
        # busy networks
        s.sendall(bytes_read)
        # update the progress bar
        progress.update(len(bytes_read))
# close the socket
s.close()


