import os
import csv
import tensorflow as tf
import pandas as pd

image_height = 600
image_width = 600

def read_image(image_file, label):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_image(image, channels=3, dtype=tf.float32)
    return image, label

class DatasetPreparer:
    def __init__(self, csv_file_name_arg: str = ""):
        self.csv_file_name = csv_file_name_arg
        self.csv_file_full_path = ""
        self.find_some_csv_file(self.csv_file_name)
        self.data_frame = pd.read_csv(self.csv_file_full_path)
        print(self.data_frame)
        self.data_paths = self.data_frame["file_name"].values
        self.labels = self.data_frame["label"].values
        self.dataset = tf.data.Dataset.from_tensor_slices((self.data_paths, self.labels))
        self.dataset = self.dataset.map(read_image).batch(1)
        print(self.dataset)
        print(len(self.dataset))
        for data in self.dataset:
            print(data[0])

    def find_some_csv_file(self, csv_file_name: str = ""):
        file_found = False
        for r,d,f in os.walk("/home/"):
            for files in f:
                if files == csv_file_name:
                    print(os.path.join(r,files))
                    self.csv_file_full_path = os.path.join(r,files)
                    file_found = True
                    break
                else:
                    continue
            if file_found:
                break
            else:
                continue
            

if __name__ == "__main__":
    dataset_preparer_obj = DatasetPreparer(csv_file_name_arg="labels_600x600_rgb_train.csv")

    
