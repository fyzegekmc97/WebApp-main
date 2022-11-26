import numpy
import cv2
import numpy as np
import glob

image_width = 28
image_height = 28


def main():
    file_list_paths = glob.glob("images/*.jpg")
    file_list = []
    for i in range(len(file_list_paths)):
        file_list.append(file_list_paths[i].split("/")[-1])

    for i in range(len(file_list_paths)):
        print(file_list[i])

    greyscale_image_tensor = np.ndarray(shape=(6, image_height, image_width))
    greyscale_image_tensor.fill(0)
    label_tensor = np.ndarray((6, 1), int)
    label_tensor.fill(1)
    for i in range(len(file_list_paths)):
        image_original = cv2.imread(file_list_paths[i])
        image_resized = cv2.resize(image_original, (image_width, image_height))
        dest = 0
        image_resized_greyscale = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY, dest)
        print(image_resized_greyscale.shape)
        cv2.waitKey(1)
        greyscale_image_tensor[i, :, :] = image_resized_greyscale

    np.save("custom_x_train.npy", greyscale_image_tensor)
    np.save("custom_y_train.npy", label_tensor)
    return file_list_paths, file_list


if __name__ == '__main__':
    main()
