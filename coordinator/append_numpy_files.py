import os
import pickle
import time
from unittest import result
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
    np.save(file=(keyword_to_contain + "_combined_coordinator.npy"), arr=result_numpy_array, allow_pickle=True)
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
    np.save(file=(keyword_to_contain + "_combined_coordinator.npy"), arr=result_numpy_array, allow_pickle=True)
    print(result_numpy_array.shape)


def append_certain_amount_of_image_files(keyword_to_contain="images", file_extension="npy", numpy_file_count: int = 5):
    curr_dir = os.curdir
    curr_dir_files_and_folders = os.listdir(curr_dir)
    res = []
    for file in os.listdir(curr_dir):
        if file.endswith(file_extension) and keyword_to_contain in file and "combined" not in file:
            res.append(file)
    result_numpy_array = np.ndarray(shape=(1, width, height, 3))
    for i in range(numpy_file_count):
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
    np.save(file=(keyword_to_contain + "_combined_coordinator.npy"), arr=result_numpy_array, allow_pickle=True)
    print(result_numpy_array.shape)


def append_certain_amount_of_label_files(keyword_to_contain="labels", file_extension="npy", numpy_file_count: int = 5):
    curr_dir = os.curdir
    curr_dir_files_and_folders = os.listdir(curr_dir)
    res = []
    for file in os.listdir(curr_dir):
        if file.endswith(file_extension) and keyword_to_contain in file and "combined" not in file:
            res.append(file)
    result_numpy_array = np.ndarray(shape=(1,))
    for i in range(numpy_file_count):
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
    np.save(file=(keyword_to_contain + "_combined_coordinator.npy"), arr=result_numpy_array, allow_pickle=True)
    print(result_numpy_array.shape)


def append_image_files_with_test_train_split(keyword_to_contain="images", file_extension="npy", training_data_percentage=60):
    curr_dir = os.curdir
    curr_dir_files_and_folders = os.listdir(curr_dir)
    res = []
    for file in os.listdir(curr_dir):
        if file.endswith(file_extension) and keyword_to_contain in file and "combined" not in file:
            res.append(file)
    result_numpy_array = np.ndarray(shape=(1, width, height, 3))
    training_data_count = int(len(res)*training_data_percentage/100)
    testing_data_count = len(res) - training_data_count
    for i in range(training_data_count):
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
    np.save(file=(keyword_to_contain + "_combined_coordinator_train.npy"), arr=result_numpy_array, allow_pickle=True)
    print(result_numpy_array.shape)
    result_numpy_array = np.ndarray(shape=(1, width, height, 3))
    for i in range(testing_data_count):
        try:
            temp_array = np.load(file=res[training_data_count + i])
        except ValueError:
            print("Iteration: ", i, "\nFile name: ", res[training_data_count + i])
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
    np.save(file=(keyword_to_contain + "_combined_coordinator_test.npy"), arr=result_numpy_array, allow_pickle=True)
    print(result_numpy_array.shape)


def append_image_files_with_test_train_split_from_certain_amount_of_files(keyword_to_contain="images", file_extension="npy", training_data_percentage=50, total_files_to_split=3):
    curr_dir = os.curdir
    curr_dir_files_and_folders = os.listdir(curr_dir)
    curr_dir_files_and_folders.sort()
    res = []
    file_count = 0
    for file in curr_dir_files_and_folders:
        if file.endswith(file_extension) and keyword_to_contain in file and "combined" not in file:
            res.append(file)
            file_count += 1
        if file_count >= total_files_to_split:
            break
    result_numpy_array = np.ndarray(shape=(1, width, height, 3))
    training_data_count = int(len(res)*training_data_percentage/100)
    print(training_data_count)
    testing_data_count = len(res) - training_data_count
    print(testing_data_count)
    print(res)
    for i in range(training_data_count):
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
    np.save(file=(keyword_to_contain + "_combined_coordinator_train.npy"), arr=result_numpy_array, allow_pickle=True)
    print(result_numpy_array.shape)
    result_numpy_array = np.ndarray(shape=(1, width, height, 3))
    for i in range(testing_data_count):
        try:
            temp_array = np.load(file=res[training_data_count + i])
        except ValueError:
            print("Iteration: ", i, "\nFile name: ", res[training_data_count + i])
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
    np.save(file=(keyword_to_contain + "_combined_coordinator_test.npy"), arr=result_numpy_array, allow_pickle=True)
    print(result_numpy_array.shape)


def append_label_files_with_test_train_split_from_certain_amount_of_files(keyword_to_contain="labels", file_extension="npy", training_data_percentage=50, total_files_to_split=3):
    curr_dir = os.curdir
    curr_dir_files_and_folders = os.listdir(curr_dir)
    curr_dir_files_and_folders.sort()
    res = []
    file_count = 0
    for file in curr_dir_files_and_folders:
        if file.endswith(file_extension) and keyword_to_contain in file and "combined" not in file:
            res.append(file)
            file_count += 1
        if file_count >= total_files_to_split:
            break
    result_numpy_array = np.ndarray(shape=(1,))
    training_count = int(len(res)*training_data_percentage/100)
    print(training_count)
    test_count = len(res) - training_count
    print(test_count)
    print(res)
    for i in range(training_count):
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
    np.save(file=(keyword_to_contain + "_combined_coordinator_train.npy"), arr=result_numpy_array, allow_pickle=True)
    print(result_numpy_array.shape)
    result_numpy_array = np.ndarray(shape=(1,))
    for i in range(test_count):
        temp_array = np.load(file=res[training_count + i])
        if i == 0:
            result_numpy_array[0] = temp_array[0]
            for k in range(1, temp_array.shape[0]):
                temp_label_array = temp_array[k, :]
                result_numpy_array = np.append(arr=result_numpy_array, values=temp_label_array, axis=0)
        else:
            for k in range(0, temp_array.shape[0]):
                temp_label_array = temp_array[k, :]
                result_numpy_array = np.append(arr=result_numpy_array, values=temp_label_array, axis=0)
    np.save(file=(keyword_to_contain + "_combined_coordinator_test.npy"), arr=result_numpy_array, allow_pickle=True)
    print(result_numpy_array.shape)


def split_all_image_data_train_test_grayscale(keyword_to_contain="images", file_extension="npy", training_data_percentage=70, total_files_to_split=11):
    curr_dir = os.curdir
    curr_dir_files_and_folders = os.listdir(curr_dir)
    curr_dir_files_and_folders.sort()
    res = []
    file_count = 0
    for file in curr_dir_files_and_folders:
        if file.endswith(file_extension) and keyword_to_contain in file and "combined" not in file and "testing" not in file and "training" not in file:
            res.append(file)
            file_count += 1
        if file_count >= total_files_to_split:
            break
    result_numpy_array = np.load(res[0])
    print(len(res))
    print(res)
    for i in range(1,len(res)):
        result_numpy_array = np.concatenate([result_numpy_array, np.load(res[i])])
    print(result_numpy_array.shape)
    training_data_count = int(result_numpy_array.shape[0]*training_data_percentage/100)
    test_data_count = int(result_numpy_array.shape[0] - training_data_count)
    np.save((keyword_to_contain + str(training_data_percentage) + "_percent_training_data.npy"), result_numpy_array[0:training_data_count])
    np.save((keyword_to_contain + str(100 - training_data_percentage) + "_percent_testing_data.npy"), result_numpy_array[training_data_count:])


def split_all_label_data_train_test_grayscale(keyword_to_contain="labels", file_extension="npy", training_data_percentage=70, total_files_to_split=11):
    curr_dir = os.curdir
    curr_dir_files_and_folders = os.listdir(curr_dir)
    curr_dir_files_and_folders.sort()
    res = []
    file_count = 0
    for file in curr_dir_files_and_folders:
        if file.endswith(file_extension) and keyword_to_contain in file and "combined" not in file and "testing" not in file and "training" not in file:
            res.append(file)
            file_count += 1
        if file_count >= total_files_to_split:
            break
    result_numpy_array = np.load(res[0])
    result_numpy_array = np.load(res[0])
    print(len(res))
    print(res)
    for i in range(1,len(res)):
        result_numpy_array = np.concatenate([result_numpy_array, np.load(res[i])])
    print(result_numpy_array.shape)
    training_data_count = int(result_numpy_array.shape[0]*training_data_percentage/100)
    test_data_count = int(result_numpy_array.shape[0] - training_data_count)
    np.save((keyword_to_contain + str(training_data_percentage) + "_percent_training_data.npy"), result_numpy_array[0:training_data_count])
    np.save((keyword_to_contain + str(100 - training_data_percentage) + "_percent_testing_data.npy"), result_numpy_array[training_data_count:])

if __name__ == "__main__":
    split_all_image_data_train_test_grayscale()
    split_all_label_data_train_test_grayscale()
    pass
    
