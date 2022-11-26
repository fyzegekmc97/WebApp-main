from cv2 import resize
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import os
import sys


target_path = "/home/feyzi/Packages/GitHubPackages/WebApp-main/WebApp-main/coordinator/NO20220510-080636-000008F/"
images_to_read = os.listdir(target_path)
index = 0


while True:
    some_image = cv2.imread(target_path + images_to_read[index], cv2.IMREAD_COLOR)
    resized = cv2.resize(some_image, dsize=(200,200))
    cv2.imshow("Resized", resized)
    pressed = cv2.waitKey(1)
    if pressed == ord("q"):
        cv2.destroyAllWindows()
        break
    elif pressed == ord("n"):
        if index + 1 == len(images_to_read):
            continue
        else:
            index += 1
            continue
    elif pressed == ord("p"):
        if index - 1 == 0:
            continue
        else:
            index -= 1
            continue
    elif pressed == ord("e"):
        index = len(images_to_read) - 1
        continue