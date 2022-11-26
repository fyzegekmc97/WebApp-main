import numpy as np
import os
import PIL
from PIL import Image
import tensorflow as tf
import tensorflow_datasets as tfds
import pathlib


def get_dataset_from_tensorflow_api() -> None:
    dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    data_dir = tf.keras.utils.get_file(origin=dataset_url,
                                   fname='flower_photos',
                                   untar=True)
    data_dir = pathlib.Path(data_dir)
    print(data_dir)
    image_count = len(list(data_dir.glob('*/*.jpg')))
    print(image_count)
    batch_size = 32
    img_height = 180
    img_width = 180
    train_ds = tf.keras.utils.image_dataset_from_directory(data_dir, validation_split=0.2, subset="training", seed=123, image_size=(img_height, img_width), batch_size=batch_size)
    print(train_ds)


def reduce_image_size(image_path: str = "", directory: str = "" , list_of_files: list[str] = [], file_extension: str = "") -> None:
    if file_extension == "":
        file_extension = ".png"
    if len(list_of_files) == 0 and image_path == "" and directory == "": # ignore image_path variable
        directory = os.curdir
        image_path = ""
        list_of_files = find_files_with_extension(directory=directory, extension=file_extension)
    elif directory != "":
        list_of_files = find_files_with_extension(directory=directory, extension=file_extension)
    
    if len(list_of_files) > 0:
        for i in range(len(list_of_files)):
            print(list_of_files[i])
            image = Image.open(os.path.join(directory, list_of_files[i]))
            image.save(os.path.join(directory, list_of_files[i]), optimize=True, quality=85)
    else:
        image = Image.open(image_path)
        image.save(image_path, optimize=True, quality=85)
        pass


def find_files_with_extension(directory: str = "", extension = "", list_of_files: list[str] = []) -> list[str]:
    current_directory = os.curdir
    target_directory = ""
    target_extension = ""
    if directory == "":
        target_directory = current_directory
    else:
        target_directory = directory
    if extension == "":
        target_extension = ".png"
    else:
        target_extension = extension
    files_and_folders_in_target_directory = os.listdir(target_directory)
    files_with_extension = []
    if len(list_of_files) == 0:
        for i in range(len(files_and_folders_in_target_directory)):
            if extension in files_and_folders_in_target_directory[i]:
                files_with_extension.append(files_and_folders_in_target_directory[i])
    else:
        for i in range(len(list_of_files)):
            if extension in list_of_files[i]:
                files_with_extension.append(list_of_files[i])
    return files_with_extension





if __name__ == "__main__":
    # reduce_image_size()
    # get_dataset_from_tensorflow_api()
    print(os.listdir(os.curdir))
