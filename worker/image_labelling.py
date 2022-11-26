import json
import cv2
import os
import matplotlib.pyplot as plt
from video_to_images import Video_To_Image_Converter
from typing import List, Optional, Any
import numpy as np
from PIL import Image
from math import ceil


class Labeller:
    def __init__(self, jpg_files_directory_arg: Optional[str] = None, jpg_file_list_arg: Optional[List[str]] = None, labels_npy_file_location_arg: Optional[str] = None, images_npy_file_location_arg: Optional[str] = None, numpy_file_name_suffix_arg: str = "NO20220510-080636-000010F"):
        self.current_index = int(0)
        self.jpg_file_list = jpg_file_list_arg
        self.jpg_files_directory = str()
        self.numpy_file_name_suffix = numpy_file_name_suffix_arg
        if jpg_files_directory_arg is None:
            self.jpg_files_directory = "images"
        else:
            self.jpg_files_directory = jpg_files_directory_arg
            self.jpg_files_directory.removesuffix("/")
            self.jpg_files_directory.removeprefix(".")
            self.jpg_files_directory.removeprefix("/")

        if self.jpg_file_list is None or len(self.jpg_file_list) == 0:
            try:
                self.jpg_file_list = os.listdir(self.jpg_files_directory)
            except FileNotFoundError:
                os.mkdir(self.jpg_files_directory)
                self.jpg_file_list = os.listdir(self.jpg_files_directory)
        self.filter_file_list()
        self.jpg_file_list.sort()
        print(self.jpg_file_list)
        self.total_files_to_label = len(self.jpg_file_list)
        self.labels_dict = dict()
        self.labels = np.ndarray(shape=(1, self.total_files_to_label))
        self.labels.fill(2.0)
        self.labels_json_obj = json.dumps("")
        self.json_file_name = "labels_json.json"
        self.json_file = open(self.json_file_name, "a+")
        self.json_file.truncate(0)
        self.image_width = 200
        self.image_height = 200
        self.images_numpy_array = Any  # np.ndarray(shape=(len(self.jpg_file_list), self.image_height, self.image_width))
        self.labels_numpy_array = Any  # np.ndarray(shape=(len(self.jpg_file_list), 1))
        self.images_npy_file_location = str()
        if images_npy_file_location_arg is None:
            self.images_npy_file_location = "images" + self.numpy_file_name_suffix + ".npy"
        else:
            self.images_npy_file_location = images_npy_file_location_arg
        self.labels_npy_file_location = str()
        if labels_npy_file_location_arg is None:
            self.labels_npy_file_location = "labels" + self.numpy_file_name_suffix + ".npy"
        else:
            self.labels_npy_file_location = labels_npy_file_location_arg
        self.images_npy_file_handle = open(self.images_npy_file_location, "a+")
        self.labels_npy_file_handle = open(self.labels_npy_file_location, "a+")
        self.images_npy_file_handle.truncate(0)
        self.labels_npy_file_handle.truncate(0)
        self.quit_key_is_pressed = False
        self.last_image_reached = False

    def __del__(self):
        self.images_npy_file_handle.close()
        self.json_file.close()
        self.labels_npy_file_handle.close()

    def filter_file_list(self, file_format=".jpg"):
        if type(self.jpg_file_list) is List[str] or type(self.jpg_file_list) is list:
            index = 0
            curr_list_len = len(self.jpg_file_list)
            while index < curr_list_len:
                if file_format in self.jpg_file_list[index]:
                    index += 1
                    continue
                else:
                    self.jpg_file_list.remove(self.jpg_file_list[index])
                    curr_list_len = len(self.jpg_file_list)
            self.total_files_to_label = len(self.jpg_file_list)
        else:
            if file_format in self.jpg_file_list:
                self.total_files_to_label = 1
                pass
            else:
                self.jpg_file_list = os.listdir(os.curdir)
                self.total_files_to_label = len(self.jpg_file_list)

    def print_field_values(self):
        object_fields = vars(self)
        object_field_names = list(object_fields.keys())
        object_field_values = list(object_fields.values())
        for i in range(len(object_field_names)):
            print("Value stored in field ", object_field_names[i], " is ", object_field_values[i], " of type ", type(object_field_values[i]))

    def label_values(self):
        while not self.quit_key_is_pressed and self.current_index < self.total_files_to_label:
            image_current = cv2.imread(os.curdir + "/" + self.jpg_files_directory + "/" + self.jpg_file_list[self.current_index])
            print("Labelling: ", os.curdir + "/" + self.jpg_files_directory + "/" + self.jpg_file_list[self.current_index])
            image_current = cv2.resize(image_current, dsize=(self.image_width, self.image_height))
            print("Labelling image with shape: ", image_current.shape)
            cv2.imshow("Current Image", image_current)
            pressed_key = cv2.waitKey(0)
            if pressed_key == ord("q"):
                cv2.destroyAllWindows()
                print("Application exit request obtained")
                print("Labelled images and the labels are: ", self.labels_dict)
                labels_keys = list(self.labels_dict.keys())
                print(labels_keys)
                self.images_numpy_array = np.ndarray(shape=(len(self.labels_dict.keys()), self.image_height, self.image_width, 3))
                self.labels_numpy_array = np.ndarray(shape=(len(self.labels_dict.keys()), 1))
                for i in range(len(self.labels_dict.keys())):
                    print("Iteration: ", i + 1)
                    print("Labeled image: ", labels_keys[i])
                    print("Original image size is: ", os.stat(os.curdir + "/" + self.jpg_files_directory + "/" + labels_keys[i]).st_size)
                    image = Image.open(os.curdir + "/" + self.jpg_files_directory + "/" + labels_keys[i])
                    image.save(os.curdir + "/" + self.jpg_files_directory + "/" + labels_keys[i], quality=15)
                    print("New image size is: ", os.stat(os.curdir + "/" + self.jpg_files_directory + "/" + labels_keys[i]).st_size)
                    returned_image_matrix = cv2.imread(os.curdir + "/" + self.jpg_files_directory + "/" + labels_keys[i])
                    returned_image_matrix = np.resize(returned_image_matrix, (self.image_height, self.image_width, 3))
                    self.images_numpy_array[i, :, :, :] = returned_image_matrix
                    self.labels_numpy_array[i, 0] = self.labels_dict[labels_keys[i]]
                    print("Image's label: ", self.labels_dict[labels_keys[i]])
                    self.labels[0][i] = self.labels_dict[labels_keys[i]]
                self.labels_json_obj = json.dumps(self.labels_dict)
                self.json_file.write(self.labels_json_obj)
                self.json_file.close()
                np.save(self.images_npy_file_location, self.images_numpy_array)
                np.save(self.labels_npy_file_location, self.labels_numpy_array)
                self.images_npy_file_handle.close()
                self.labels_npy_file_handle.close()
                print("Numeric labels are: ", self.labels)
                self.quit_key_is_pressed = True
                break
            elif pressed_key == ord("n"):
                cv2.destroyAllWindows()
                if self.current_index + 1 == self.total_files_to_label:
                    self.last_image_reached = True
                    break
                self.current_index += 1
                print("Going to next image")
            elif pressed_key == ord("p"):
                cv2.destroyAllWindows()
                self.current_index -= 1
                print("Going to previous image")
            elif pressed_key == ord("k"):
                print("File with name ", self.jpg_file_list[self.current_index], " is confirmed to have a pedestrian")
                self.labels_dict[self.jpg_file_list[self.current_index]] = 1
            elif pressed_key == ord("l"):
                print("File with name ", self.jpg_file_list[self.current_index], " is confirmed to NOT have a pedestrian")
                self.labels_dict[self.jpg_file_list[self.current_index]] = 0
            elif pressed_key == ord("e"):
                print("Going to last image")
                self.current_index = self.total_files_to_label - 1
            else:
                print(pressed_key)

        if self.last_image_reached:
            cv2.destroyAllWindows()
            print("Application exit request obtained")
            print("Labelled images and the labels are: ", self.labels_dict)
            labels_keys = list(self.labels_dict.keys())
            print(labels_keys)
            self.images_numpy_array = np.ndarray(
                shape=(len(self.labels_dict.keys()), self.image_height, self.image_width, 3))
            self.labels_numpy_array = np.ndarray(shape=(len(self.labels_dict.keys()), 1))
            for i in range(len(self.labels_dict.keys())):
                print("Iteration: ", i + 1)
                print("Labeled image: ", labels_keys[i])
                returned_image_matrix = cv2.imread(os.curdir + "/" + self.jpg_files_directory + "/" + labels_keys[i])
                returned_image_matrix = np.resize(returned_image_matrix, (self.image_height, self.image_width, 3))
                self.images_numpy_array[i, :, :, :] = returned_image_matrix
                self.labels_numpy_array[i, 0] = self.labels_dict[labels_keys[i]]
                print("Image's label: ", self.labels_dict[labels_keys[i]])
                self.labels[0][i] = self.labels_dict[labels_keys[i]]
            self.labels_json_obj = json.dumps(self.labels_dict)
            self.json_file = open(self.json_file_name, "a+")
            self.json_file.write(self.labels_json_obj)
            self.json_file.close()
            self.images_npy_file_handle = open(self.images_npy_file_location, "a+")
            self.labels_npy_file_handle = open(self.labels_npy_file_location, "a+")
            np.save(self.images_npy_file_location, self.images_numpy_array)
            np.save(self.labels_npy_file_location, self.labels_numpy_array)
            self.images_npy_file_handle.close()
            self.labels_npy_file_handle.close()
            print("Numeric labels are: ", self.labels)

        saved_image_array = np.load(self.images_npy_file_location)
        saved_label_array = np.load(self.labels_npy_file_location)
        print(saved_image_array.shape)
        print(saved_label_array.shape)


class Renamer:
    def __init__(self, input_directory_arg: str = os.curdir, target_directory_arg: str = os.curdir, file_extension_arg: str = "mp4"):
        self.current_directory = os.curdir
        self.input_directory = input_directory_arg
        self.target_directory = target_directory_arg
        self.file_extension = file_extension_arg
        if "." in self.file_extension:
            self.file_extension = self.file_extension[self.file_extension.rfind(".")+1:]
        self.current_directory_files = os.listdir(self.current_directory)
        self.input_directory_files = os.listdir(self.input_directory)
        index = 0
        file_list_length = len(self.input_directory_files)
        while index < file_list_length:
            if self.input_directory_files[index].endswith(self.file_extension):
                if "combined" in self.input_directory_files[index]:
                    self.input_directory_files.remove(self.input_directory_files[index])
                    file_list_length = len(self.input_directory_files)
                    continue
                else:
                    index += 1
                    continue
            else:
                self.input_directory_files.remove(self.input_directory_files[index])
                file_list_length = len(self.input_directory_files)
                pass
        print(self.input_directory_files)
        for i in range(len(self.input_directory_files)):
            file_name_split = self.input_directory_files[i].split(".")
            new_file_name = str(i) + "_worker_" + "."  + file_name_split[-1]
            os.rename(self.input_directory_files[i], new_file_name)
            self.input_directory_files[i] = new_file_name
        print(self.input_directory_files)


if __name__ == "__main__":
    """ converter_obj = Video_To_Image_Converter(["NO20220510-124353-000286F.mp4"], images_target_directory_arg="NO20220510-124353-000286F/")
    converter_obj.convert_to_images()
    labeller_obj = Labeller(jpg_files_directory_arg="NO20220510-124353-000286F/", numpy_file_name_suffix_arg="NO20220510-124353-000286F")
    labeller_obj.label_values() """
    some_renamer = Renamer()
