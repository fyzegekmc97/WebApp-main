# This is server code to send video frames over UDP
from asyncio import sleep
import cv2
import imutils
import socket
import numpy as np
import time
import base64

BUFF_SIZE = 65536
server_socket = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
server_socket.setsockopt(socket.SOL_SOCKET,socket.SO_RCVBUF,BUFF_SIZE)
host_name = socket.gethostname()
host_ip = '10.147.17.7'#  socket.gethostbyname(host_name)
print(host_ip)
port = 9999
socket_address = (host_ip,port)
target_address = ('10.147.17.147', 3838)
server_socket.bind(socket_address)
print('Listening at:', socket_address)

vid = cv2.VideoCapture(0) #  replace 'rocket.mp4' with 0 for webcam
fps,st,frames_to_count,cnt = (0,0,20,0)

embe = b''

def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

while True:
	# msg,client_addr = server_socket.recvfrom(BUFF_SIZE)
	# print('GOT connection from ',client_addr)
	WIDTH=600 ##400 ile oynyınca boyut küçülüyor.
	while(vid.isOpened()):
		_,frame = vid.read()
		frame = imutils.resize(frame,width=WIDTH)
		encoded,buffer = cv2.imencode('.jpg',frame,[cv2.IMWRITE_JPEG_QUALITY,30])                       ##40 numara ile oynayınca görüntü kalitesi düşüyor.
		message = base64.b64encode(buffer)
		#server_socket.sendto(message,client_addr)

		
		ahmet = split(message,4)

	    

		print('**************************************************')

		for mes in ahmet:
			embe += mes
			server_socket.sendto(mes,target_address)
			time.sleep(0.01)
		
		print(embe)

		#time.sleep(0.1)


		
			
		embe = b''

		key = cv2.waitKey(1) & 0xFF
		if key == ord('q'):
			server_socket.close()
			break
		if cnt == frames_to_count:
			try:
				fps = round(frames_to_count/(time.time()-st))
				st=time.time()
				cnt=0
			except:
				pass
		cnt+=1
