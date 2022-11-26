import cv2
import os
from typing import Optional, Union, List
import time


class Video_To_Image_Converter:
    """
    Class for converting videos to images.
    """
    def __init__(self, input_file_list_arg: Union[List[str], str, None] = None, images_target_directory_arg: Union[str, None] = None):
        """
        Constructor for the class Video_To_Image_Converter

        :param input_file_list_arg: List of input files for converting videos to images
        :param images_target_directory_arg: The directory to save the images into
        """

        self.files_and_folders_in_current_directory = os.listdir(os.curdir)
        for i in range(len(self.files_and_folders_in_current_directory)):
            if os.path.isfile(self.files_and_folders_in_current_directory[i]):
                if self.files_and_folders_in_current_directory[i].endswith(".MP4"):
                    old_name = self.files_and_folders_in_current_directory[i]
                    self.files_and_folders_in_current_directory[i] = self.files_and_folders_in_current_directory[i].removesuffix(".MP4")
                    self.files_and_folders_in_current_directory[i] = self.files_and_folders_in_current_directory[
                                                                         i] + ".mp4"
                    new_name = self.files_and_folders_in_current_directory[i]
                    os.rename(old_name, new_name)

        self.input_file_list = []
        if input_file_list_arg is not None:
            self.input_file_list = input_file_list_arg
        else:
            self.input_file_list = os.listdir(os.curdir)
            self.filter_file_list()

        self.images_target_directory = images_target_directory_arg
        print("Got input file list type as: ", type(self.input_file_list))
        print("Got input file list as: ", self.input_file_list)
        print("Got output images directory type as: ", type(self.images_target_directory))
        print("Got output images directory as: ", self.images_target_directory)
        if self.images_target_directory is None:
            self.images_target_directory = "images/"
            if not os.path.isdir(os.curdir + "/" + self.images_target_directory):
                os.mkdir(self.images_target_directory)
        else:
            if not os.path.isdir(os.curdir + "/" + self.images_target_directory):
                os.mkdir(self.images_target_directory)

        if type(self.input_file_list) is str:
            self.filter_file_list()
        else:
            try:
                if len(self.input_file_list) == 0:
                    self.input_file_list = os.listdir()
                    self.filter_file_list()
                    print(self.input_file_list)
            except TypeError:
                self.input_file_list = os.listdir(os.curdir)
                self.filter_file_list()


        print("At init, input file list is: ", self.input_file_list)
        print("At init, images will be saved in the following relative directory: " + "./" + self.images_target_directory)
        print("At init, images will be saved in the following absolute directory: " + os.path.abspath(".") + "/" + self.images_target_directory)


    def filter_file_list(self, file_format=".mp4"):
        if type(self.input_file_list) is List[str] or type(self.input_file_list) is list:
            index = 0
            curr_list_len = len(self.input_file_list)
            while index < curr_list_len:
                if file_format in self.input_file_list[index]:
                    index += 1
                    continue
                else:
                    self.input_file_list.remove(self.input_file_list[index])
                    curr_list_len = len(self.input_file_list)
        else:
            if file_format in self.input_file_list:
                pass
            else:
                self.input_file_list = os.listdir(os.curdir)

    def convert_to_images(self):
        begin = time.perf_counter()
        frame_count_to_skip = 30
        if type(self.input_file_list) is list:
            for i in range(len(self.input_file_list)):
                video_capture = cv2.VideoCapture(self.input_file_list[i])
                video_capture.set(cv2.CAP_PROP_FPS, 10)
                success_video_capture, image_video_capture = video_capture.read()
                image_count = 0
                print("Started reading video file ", self.input_file_list[i])
                print(os.curdir + self.images_target_directory)
                while success_video_capture:
                    if image_count % frame_count_to_skip == 0:
                        cv2.imwrite("%s%sframe%d.jpg" % (os.curdir + "/" + self.images_target_directory, self.input_file_list[i].replace(".mp4", ""), image_count), image_video_capture)
                    success_video_capture, image_video_capture = video_capture.read()
                    image_count += 1

                print("Finished converting file ", self.input_file_list[i], " imto directory ", self.images_target_directory)
        elif type(self.input_file_list) is str:
            video_capture = cv2.VideoCapture(self.input_file_list)
            video_capture.set(cv2.CAP_PROP_FPS, 10)
            success_video_capture, image_video_capture = video_capture.read()
            image_count = 0
            print("Started reading video file ", self.input_file_list)
            print(os.curdir + "/" + self.images_target_directory)
            while success_video_capture:
                if image_count % frame_count_to_skip == 0:
                    cv2.imwrite("%s%sframe%d.jpg" % (os.curdir + "/" + self.images_target_directory, str(self.input_file_list).replace(".mp4", ""), image_count), image_video_capture)
                success_video_capture, image_video_capture = video_capture.read()
                image_count += 1

        end = time.perf_counter()
        print("It took ", end - begin, " seconds for conversion")


if __name__ == "__main__":
    test_obj = Video_To_Image_Converter()
    test_obj.convert_to_images()
    print("Done")