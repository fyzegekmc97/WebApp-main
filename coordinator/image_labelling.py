import json
import cv2
import os
import sys
from video_to_images import Video_To_Image_Converter
from typing import List, Optional, Any
import numpy as np
from math import ceil
import csv
import shutil


class Labeller:
    def __init__(self, jpg_files_directory_arg: Optional[str] = None, jpg_file_list_arg: Optional[List[str]] = None, labels_npy_file_location_arg: Optional[str] = None, images_npy_file_location_arg: Optional[str] = None, numpy_file_name_suffix_arg: str = "NO20220510-080636-000010F", saved_to_common_directory_arg: bool = True, common_directory_arg: str = "photos_common_directory", jpg_files_prefix: str = "", csv_files_suffix_arg: str = "600x600_rgb_train", should_move_images_to_different_directories: bool = True, save_as_grayscale: bool = True, canny_detection_for_grayscale: bool = False):
        self.current_index = int(0)
        print(os.path.abspath(os.curdir))
        os.chdir(os.path.abspath("image_labelling.py").replace("image_labelling.py", ""))
        print(os.path.abspath(os.curdir))
        self.jpg_file_list = jpg_file_list_arg
        self.jpg_files_directory = str()
        self.csv_files_suffix = csv_files_suffix_arg
        self.numpy_file_name_suffix = numpy_file_name_suffix_arg
        self.common_directory = common_directory_arg
        self.should_save_as_grayscale = save_as_grayscale
        if os.path.exists(self.common_directory):
            pass
        else:
            os.mkdir(self.common_directory)
        if saved_to_common_directory_arg:
            self.jpg_files_directory = self.common_directory
        
        if jpg_files_directory_arg is None:
            self.jpg_files_directory = "images"
            if saved_to_common_directory_arg:
                self.jpg_files_directory = common_directory_arg
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
        if jpg_files_prefix != "":
            index = 0
            jpg_file_list_len = len(self.jpg_file_list)
            while index < jpg_file_list_len:
                if jpg_files_prefix in self.jpg_file_list[index]:
                    index += 1
                    continue
                else:
                    self.jpg_file_list.remove(self.jpg_file_list[index])
                    jpg_file_list_len = len(self.jpg_file_list)
                    continue
        self.jpg_file_list.sort()
        self.total_files_to_label = len(self.jpg_file_list)
        self.labels_dict = dict()
        self.labels = np.ndarray(shape=(1, self.total_files_to_label))
        self.labels.fill(np.Inf)
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
        self.should_move_images_to_different_directories = should_move_images_to_different_directories
        self.have_canny_detection = canny_detection_for_grayscale
        if should_move_images_to_different_directories:
            if not os.path.exists(os.path.join(os.curdir, self.jpg_files_directory, "0/")):
                os.mkdir(os.path.join(os.curdir, self.jpg_files_directory, "0/"))
            if not os.path.exists(os.path.join(os.curdir, self.jpg_files_directory, "1/")):
                os.mkdir(os.path.join(os.curdir, self.jpg_files_directory, "1/"))
        

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
            if os.path.exists(os.curdir + "/" + self.jpg_files_directory + "/" + self.jpg_file_list[self.current_index]):
                image_current = cv2.imread(os.curdir + "/" + self.jpg_files_directory + "/" + self.jpg_file_list[self.current_index])
            else:
                print("File does not exist")
            print("Labelling: ", os.curdir + "/" + self.jpg_files_directory + "/" + self.jpg_file_list[self.current_index])
            image_current = cv2.resize(image_current, dsize=(self.image_width, self.image_height))
            print("Labelling image with shape: ", image_current.shape)
            print(str(self.current_index) + "/" + str(self.total_files_to_label))
            cv2.imshow("Current Image", image_current)
            pressed_key = cv2.waitKey(0)
            if pressed_key == ord("q"):
                cv2.destroyAllWindows()
                print("Application exit request obtained")
                print("Labelled images and the labels are: ", self.labels_dict)
                labels_keys = list(self.labels_dict.keys())
                print(labels_keys)

                if self.should_save_as_grayscale:
                    self.images_numpy_array = np.ndarray(shape=(len(self.labels_dict.keys()), self.image_height, self.image_width))
                    self.labels_numpy_array = np.ndarray(shape=(len(self.labels_dict.keys()), 1))
                    if not os.path.exists("labels_" + self.csv_files_suffix + ".csv"):
                        csv_file_handle = open("labels_" + self.csv_files_suffix + ".csv", "a+", newline='')
                        csv_writer = csv.writer(csv_file_handle)
                        csv_writer.writerow(["file_name", "label"])
                    else:
                        csv_file_handle = open("labels_" + self.csv_files_suffix + ".csv", "a+", newline='')
                        csv_writer = csv.writer(csv_file_handle)
                    for i in range(len(self.labels_dict.keys())):
                        print("Iteration: ", i + 1)
                        print("Labeled image: ", labels_keys[i])
                        returned_image_matrix = cv2.imread(os.curdir + "/" + self.jpg_files_directory + "/" + labels_keys[i])
                        if self.should_move_images_to_different_directories:
                            csv_writer.writerow([os.path.abspath(os.curdir) + "/" + self.jpg_files_directory + "/" + str(self.labels_dict[labels_keys[i]]) + "/" + labels_keys[i], self.labels_dict[labels_keys[i]]])
                            shutil.move(os.curdir + "/" + self.jpg_files_directory + "/" + labels_keys[i], os.path.abspath(os.curdir) + "/" + self.jpg_files_directory + "/" + str(self.labels_dict[labels_keys[i]]) + "/" + labels_keys[i])
                        else:
                            csv_writer.writerow([os.path.abspath(os.curdir) + "/" + self.jpg_files_directory + labels_keys[i], self.labels_dict[labels_keys[i]]])
                        returned_image_matrix = cv2.resize(returned_image_matrix, (self.image_height, self.image_width), interpolation = cv2.INTER_AREA)
                        returned_image_matrix = cv2.cvtColor(returned_image_matrix, cv2.COLOR_BGR2GRAY)
                        if self.have_canny_detection:
                            img_blur = cv2.GaussianBlur(returned_image_matrix, (3,3), 0)
                            edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)
                            returned_image_matrix = edges
                        self.images_numpy_array[i, :, :] = returned_image_matrix
                        self.labels_numpy_array[i, 0] = self.labels_dict[labels_keys[i]]
                        print("Image's label: ", self.labels_dict[labels_keys[i]])
                        self.labels[0][i] = self.labels_dict[labels_keys[i]]
                else:
                    self.images_numpy_array = np.ndarray(shape=(len(self.labels_dict.keys()), self.image_height, self.image_width,3))
                    self.labels_numpy_array = np.ndarray(shape=(len(self.labels_dict.keys()), 1))
                    if not os.path.exists("labels_" + self.csv_files_suffix + ".csv"):
                        csv_file_handle = open("labels_" + self.csv_files_suffix + ".csv", "a+", newline='')
                        csv_writer = csv.writer(csv_file_handle)
                        csv_writer.writerow(["file_name", "label"])
                    else:
                        csv_file_handle = open("labels_" + self.csv_files_suffix + ".csv", "a+", newline='')
                        csv_writer = csv.writer(csv_file_handle)
                    for i in range(len(self.labels_dict.keys())):
                        print("Iteration: ", i + 1)
                        print("Labeled image: ", labels_keys[i])
                        returned_image_matrix = cv2.imread(os.curdir + "/" + self.jpg_files_directory + "/" + labels_keys[i])
                        if self.should_move_images_to_different_directories:
                            csv_writer.writerow([os.path.abspath(os.curdir) + "/" + self.jpg_files_directory + "/" + str(self.labels_dict[labels_keys[i]]) + "/" + labels_keys[i], self.labels_dict[labels_keys[i]]])
                            shutil.move(os.curdir + "/" + self.jpg_files_directory + "/" + labels_keys[i], os.path.abspath(os.curdir) + "/" + self.jpg_files_directory + "/" + str(self.labels_dict[labels_keys[i]]) + "/" + labels_keys[i])
                        else:
                            csv_writer.writerow([os.path.abspath(os.curdir) + "/" + self.jpg_files_directory + labels_keys[i], self.labels_dict[labels_keys[i]]])
                        returned_image_matrix = cv2.resize(returned_image_matrix, (self.image_height, self.image_width), interpolation = cv2.INTER_AREA)
                        self.images_numpy_array[i, :, :, :] = returned_image_matrix
                        self.labels_numpy_array[i, 0] = self.labels_dict[labels_keys[i]]
                        print("Image's label: ", self.labels_dict[labels_keys[i]])
                        self.labels[0][i] = self.labels_dict[labels_keys[i]]
                cv2.imshow("Some random image", self.images_numpy_array[0])
                self.labels_json_obj = json.dumps(self.labels_dict)
                self.json_file.write(self.labels_json_obj)
                self.json_file.close()
                np.save(self.images_npy_file_location, self.images_numpy_array)
                np.save(self.labels_npy_file_location, self.labels_numpy_array)
                self.images_npy_file_handle.close()
                self.labels_npy_file_handle.close()
                csv_file_handle.close()
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
            if not os.path.exists("labels_" + self.csv_files_suffix + ".csv"):
                csv_file_handle = open("labels_" + self.csv_files_suffix + ".csv", "a+", newline='')
                csv_writer = csv.writer(csv_file_handle)
                csv_writer.writerow(["file_name", "label"])
            else:
                csv_file_handle = open("labels_" + self.csv_files_suffix + ".csv", "a+", newline='')
                csv_writer = csv.writer(csv_file_handle)
            for i in range(len(self.labels_dict.keys())):
                print("Iteration: ", i + 1)
                print("Labeled image: ", labels_keys[i])
                returned_image_matrix = cv2.imread(os.curdir + "/" + self.jpg_files_directory + "/" + labels_keys[i])
                returned_image_matrix = cv2.resize(returned_image_matrix, (self.image_height, self.image_width), interpolation = cv2.INTER_AREA)
                csv_writer.writerow([os.path.abspath(os.curdir) + "/" + self.jpg_files_directory + labels_keys[i], self.labels_dict[labels_keys[i]]])
                returned_image_matrix = cv2.resize(returned_image_matrix, (self.image_height, self.image_width), interpolation = cv2.INTER_AREA)
                self.images_numpy_array[i, :, :, :] = returned_image_matrix
                self.labels_numpy_array[i, 0] = self.labels_dict[labels_keys[i]]
                print("Image's label: ", self.labels_dict[labels_keys[i]])
                self.labels[0][i] = self.labels_dict[labels_keys[i]]
            self.labels_json_obj = json.dumps(self.labels_dict)
            self.json_file = open(self.json_file_name, "a+")
            self.json_file.write(self.labels_json_obj)
            self.json_file.close()
            csv_file_handle.close()
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
                if "combined" in self.input_directory_files[index] or "coordinator" in self.input_directory_files[index] or "worker" in self.input_directory_files[index]:
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
            new_file_name = str(i) + "_coordinator_" + "."  + file_name_split[-1]
            os.rename(self.input_directory_files[i], new_file_name)
            self.input_directory_files[i] = new_file_name
        print(self.input_directory_files)


class Training_Image_and_Video_Data_Generator:
    def __init__(self, original_image_width_arg: int = 1080, original_image_height_arg: int = 1080, target_image_width_arg: int = 200, target_image_height_arg: int = 200, image_count_arg: int = 50, demanded_background_type_arg: str = "all_green", provided_red_component_arg: np.uint8 = 0, provided_green_component_arg: np.uint8 = 0, provided_blue_component_arg: np.uint8 = 0, provided_image_location_arg: str = "", target_image_location_arg: str = os.curdir, provided_video_location_arg: str = "", target_video_location_arg: str = "", provided_foreground_picture_location_arg: str = ""):
        self.original_image_width = original_image_width_arg
        self.original_image_height = original_image_height_arg
        self.target_image_width = target_image_width_arg
        self.target_image_height = target_image_height_arg
        self.demanded_background_type = demanded_background_type_arg
        self.provided_bgr_tuple = (provided_blue_component_arg, provided_green_component_arg, provided_red_component_arg)
        self.provided_image_location = provided_image_location_arg
        self.provided_video_location = provided_video_location_arg
        self.background_types_list = ["all_black", "all_white", "all_yellow", "all_pink", "all_purple", "all_green", "random_grayscale", "all_black_grayscale", "all_white_grayscale", "provided_rgb", "provided_image", "provided_video", "all red", "all blue"]  # Might add on more stuff here later on...
        self.foreground_types_list = ["person", "car", "bike", "stop_sign", "traffic_light", "truck", "provided_foreground_picture"]
        self.accepted_picture_extensions = ["png", "jpg", "jpeg", "tiff", "jpeg"]
        self.provided_foreground_picture_location = provided_foreground_picture_location_arg
        self.target_video_location = target_video_location_arg
        self.target_image_location = target_image_location_arg
        self.images_array = Any
        while self.demanded_background_type not in self.background_types_list:
            self.demanded_background_type = input("Please input one of the following background types: " + str(self.background_types_list))
        while self.demanded_background_type == "provided_image" and not os.path.exists(self.provided_image_location):
            self.provided_image_location = input("Image location does not exist. Provide a new background image location.")
        while self.demanded_background_type == "provided video" and not os.path.exists(self.provided_video_location):
            self.provided_video_location = input("Video location does not exist. Provide a new background video location.")
        
        some_video_reader = cv2.VideoCapture(self.provided_video_location)
        if self.demanded_background_type == self.background_types_list[11]:
            success, frame = some_video_reader.read()
            print(success)
            print(frame)
            
        
        self.image_count = image_count_arg
        self.image_to_return = Any
        self.original_image = Any
        if "grayscale" in self.demanded_background_type:
            self.original_image = np.ndarray(shape=(self.target_image_height, self.target_image_width), dtype=np.uint8)
            self.image_to_return = np.ndarray(shape=(self.target_image_height, self.target_image_width), dtype=np.uint8)
        else:
            self.original_image = np.ndarray(shape=(self.target_image_height, self.target_image_width, 3), dtype=np.uint8)
            self.image_to_return = np.ndarray(shape=(self.target_image_height, self.target_image_width, 3), dtype=np.uint8)


    def generate_image(self):
        for i in range(self.image_count):
            if self.demanded_background_type == self.background_types_list[0]:  # All black RGB
                self.original_image = np.ndarray(shape=(self.original_image_height, self.original_image_width, 3), dtype=np.uint8)
                self.original_image.fill(0)
            elif self.demanded_background_type == self.background_types_list[1]:  # All white RGB
                self.original_image = np.ndarray(shape=(self.original_image_height, self.original_image_width, 3), dtype=np.uint8)
                self.original_image.fill(255)
            elif self.demanded_background_type == self.background_types_list[2]:  # All yellow RGB
                self.original_image = np.ndarray(shape=(self.original_image_height, self.original_image_width, 3), dtype=np.uint8)
                for i in range(self.original_image_height):
                    for k in range(self.original_image_width):
                        self.original_image[i, k, :] = [0,255,255]
                cv2.imshow("All yellow", self.original_image)
                cv2.waitKey(1)
            elif self.demanded_background_type == self.background_types_list[3]:  # All pink RGB 
                self.original_image = np.ndarray(shape=(self.original_image_height, self.original_image_width, 3), dtype=np.uint8)
                for i in range(self.original_image_height):
                    for k in range(self.original_image_width):
                        self.original_image[i, k, :] = [203, 192, 255]
                cv2.imshow("All pink", self.original_image)
                cv2.waitKey(1)
                pass
            elif self.demanded_background_type == self.background_types_list[4]:  # All purple RGB
                """ 255,0,255 """
                self.original_image = np.ndarray(shape=(self.original_image_height, self.original_image_width, 3), dtype=np.uint8)
                for i in range(self.original_image_height):
                    for k in range(self.original_image_width):
                        self.original_image[i, k, :] = [255, 0, 255]
                cv2.imshow("All purple", self.original_image)
                cv2.waitKey(1)
                pass
            elif self.demanded_background_type == self.background_types_list[5]:  # All green RGB
                self.original_image = np.ndarray(shape=(self.original_image_height, self.original_image_width, 3), dtype=np.uint8)
                for i in range(self.original_image_height):
                    for k in range(self.original_image_width):
                        self.original_image[i, k, :] = [0, 128, 0]
                cv2.imshow("All green", self.original_image)
                cv2.waitKey(1)
                pass
            elif self.demanded_background_type == self.background_types_list[6]:  # Random greyscale
                self.original_image = np.ndarray(shape=(self.original_image_height, self.original_image_width), dtype=np.uint8)
                for i in range(self.original_image_height):
                    for k in range(self.original_image_width):
                        self.original_image[i, k] = np.random.randint(0, 255)
                cv2.imshow("Random grayscale", self.original_image)
                cv2.waitKey(1)
                pass
            elif self.demanded_background_type == self.background_types_list[7]: # All black grayscale
                self.original_image = np.ndarray(shape=(self.original_image_height, self.original_image_width), dtype=np.uint8)
                for i in range(self.original_image_height):
                    for k in range(self.original_image_width):
                        self.original_image[i, k] = 0
                cv2.imshow("All black grayscale", self.original_image)
                cv2.waitKey(1)
                pass
            elif self.demanded_background_type == self.background_types_list[8]:  # All white greyscale
                self.original_image = np.ndarray(shape=(self.original_image_height, self.original_image_width), dtype=np.uint8)
                for i in range(self.original_image_height):
                    for k in range(self.original_image_width):
                        self.original_image[i, k] = 255
                cv2.imshow("All white grayscale", self.original_image)
                cv2.waitKey(1)
                pass
            elif self.demanded_background_type == self.background_types_list[9]:  # Provided RGB color
                self.original_image = np.ndarray(shape=(self.original_image_height, self.original_image_width,3), dtype=np.uint8)
                for i in range(self.original_image_height):
                    for k in range(self.original_image_width):
                        for l in range(3):
                            self.original_image[i,k,l] = self.provided_bgr_tuple[l]
                cv2.imshow("Provided RGB", self.original_image)
                cv2.waitKey(1)
            elif self.demanded_background_type == self.background_types_list[10]:  # Provided image
                self.original_image = cv2.imread(self.provided_image_location, cv2.IMREAD_COLOR)
                some_tuple = self.original_image.shape
                self.original_image_height = some_tuple[0]
                self.original_image_width = some_tuple[1]
                print(self.original_image_height, self.original_image_width)
                

class LargeCaseToSmallCase:
    def __init__(self, file_extension_arg: str = "MP4") -> None:
        self.current_directory = os.curdir
        self.files_in_current_directory = os.listdir(self.current_directory)
        self.file_extension_to_look_for = file_extension_arg
        for i in range(len(self.files_in_current_directory)):
            if self.file_extension_to_look_for in self.files_in_current_directory[i]:
                temp_file_name = self.files_in_current_directory[i].lower()
                os.rename(self.files_in_current_directory[i], temp_file_name)
            else:
                continue
        self.files_in_current_directory = os.listdir(self.current_directory)


class FileMover:
    def __init__(self, directory_to_move_from: str = "", list_of_files_to_move: list[str] = [], file_to_move: str = "", target_directory: str = ""):
        if len(list_of_files_to_move) > 0:
            self.list_of_files = list_of_files_to_move
            for i in range(len(self.list_of_files)):
                if os.path.exists(self.list_of_files[i]):
                    shutil.move(self.list_of_files[i], target_directory)
        elif directory_to_move_from != "":
            self.list_of_files = os.listdir(directory_to_move_from)
            for i in range(len(self.list_of_files)):
                if os.path.exists(self.list_of_files[i]):
                    shutil.move(self.list_of_files[i], target_directory)
        elif file_to_move != "":
            shutil.move(file_to_move, target_directory)
            pass
        else:
            pass        


class DatasetFeeder:
    def __init__(self, should_pick_from_common_directory_arg: bool = True, should_pick_from_seperated_directories_arg: bool = False, common_directory_arg: str = ".", seperated_directories_location_arg: str = "."):
        self.should_pick_from_common_directory = should_pick_from_common_directory_arg
        self.should_pick_from_seperated_directories = should_pick_from_seperated_directories_arg
        self.common_directory = common_directory_arg
        self.seperated_directories_location = seperated_directories_location_arg
        self.total_files_in_dataset = 0
        self.files_in_dataset = []
        self.seperated_directories = []
        self.seperated_directories_location_dirs_and_files = os.listdir(self.seperated_directories_location)
        self.all_seperated_files = []
        if self.should_pick_from_common_directory:
            self.files_in_dataset = os.listdir(self.common_directory)
            self.total_files_in_dataset = len(self.files_in_dataset)
            print(self.total_files_in_dataset)
        elif self.should_pick_from_seperated_directories:
            root_file_name = ""
            for root, dirs, files in os.walk(self.seperated_directories_location):
                for directory in dirs:
                    self.seperated_directories.append(directory)
            
            for i in range(len(self.seperated_directories)):
                current_seperated_directory_length = len(os.listdir(os.path.join(os.path.abspath(self.seperated_directories_location), self.seperated_directories[i])))
                self.all_seperated_files.extend(os.listdir(os.path.join(os.path.abspath(self.seperated_directories_location), self.seperated_directories[i])))
                print(current_seperated_directory_length)

            print(self.all_seperated_files)


if __name__ == "__main__":
    labeller_obj = Labeller(numpy_file_name_suffix_arg="_no20220719-125005-000003f_200_200_rgb_",saved_to_common_directory_arg=True, jpg_files_prefix="no20220719-125005-000003f", csv_files_suffix_arg="200x200_rgb")
    labeller_obj.label_values()
    pass
