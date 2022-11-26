import os
import pickle
import time
import numpy as np

width = 200
height = 200


def sandbox_test():
    curr_time_ns_of_day = int(time.time() % (24 * 60 * 60 * 10000000))
    np.random.seed(curr_time_ns_of_day)
    some_random_array = np.random.random(size=(10, 10))
    print(some_random_array)


def append_images(keyword_to_contain="images", file_extension="npy"):
    curr_dir = os.curdir
    curr_dir_files_and_folders = os.listdir(curr_dir)
    res = []
    for file in os.listdir(curr_dir):
        if file.endswith(file_extension) and keyword_to_contain in file and "combined" not in file:
            res.append(file)
    result_numpy_array = np.ndarray(shape=(1, width, height, 3))
    for i in range(len(res)):
        temp_array = np.load(file=res[i])
        if i == 0:
            temp_image_array = temp_array[0, :, :, :]
            temp_image_array = np.resize(a=temp_image_array, new_shape=(width, height, 3))
            temp_image_array = np.reshape(temp_image_array, (1, width, height, 3))
            result_numpy_array[0, :, :, :] = temp_image_array
            for k in range(1, temp_array.shape[0]):
                temp_image_array = temp_array[k, :, :, :]
                temp_image_array = np.resize(a=temp_image_array, new_shape=(width, height, 3))
                temp_image_array = np.reshape(temp_image_array, (1, width, height, 3))
                result_numpy_array = np.append(arr=result_numpy_array, values=temp_image_array, axis=0)
        else:
            for k in range(0, temp_array.shape[0]):
                temp_image_array = temp_array[k, :, :, :]
                temp_image_array = np.resize(a=temp_image_array, new_shape=(width, height, 3))
                temp_image_array = np.reshape(temp_image_array, (1, width, height, 3))
                result_numpy_array = np.append(arr=result_numpy_array, values=temp_image_array, axis=0)
    np.save(file=(keyword_to_contain + "_combined_worker.npy"), arr=result_numpy_array)
    print(result_numpy_array.shape)


def append_labels(keyword_to_contain="labels", file_extension="npy"):
    curr_dir = os.curdir
    curr_dir_files_and_folders = os.listdir(curr_dir)
    res = []
    for file in os.listdir(curr_dir):
        if file.endswith(file_extension) and keyword_to_contain in file and "combined" not in file:
            res.append(file)
    result_numpy_array = np.ndarray(shape=(1,))
    for i in range(len(res)):
        temp_array = np.load(file=res[i])
        if i == 0:
            result_numpy_array[0] = temp_array[0]
            for k in range(1, temp_array.shape[0]):
                temp_label_array = temp_array[k, :]
                result_numpy_array = np.append(arr=result_numpy_array, values=temp_label_array, axis=0)
        else:
            for k in range(0, temp_array.shape[0]):
                temp_label_array = temp_array[k, :]
                result_numpy_array = np.append(arr=result_numpy_array, values=temp_label_array, axis=0)
    np.save(file=(keyword_to_contain + "_combined_worker.npy"), arr=result_numpy_array)
    print(result_numpy_array.shape)


if __name__ == "__main__":
    append_images()
    append_labels()
