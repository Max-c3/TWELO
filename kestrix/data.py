import os
import random
from pathlib import Path
import pandas as pd
import tensorflow as tf
from google.cloud import storage
from tqdm import tqdm

from twelo.params import *


def download_raw_data():
    """Download raw data from google cloud to `data/twelo/raw/`.

    Returns:
        None
    """
    remote_path = "data/raw/"
    local_path_full = "data/twelo/raw/"

    client = storage.Client()
    blobs = client.list_blobs(BUCKET_NAME, prefix=remote_path, delimiter="/")

    if os.path.exists(local_path_full):
        print("Dataset already exists.")
    else:
        os.makedirs(local_path_full)
        print(f"Downloading dataset to {local_path_full}.")
        for blob in tqdm(blobs):
            file_name = Path(blob.name).name
            blob.download_to_filename(local_path_full + file_name)
        print("Finished download.")

    return None

# def download_comp_data():
#     """Download compartmented data from google cloud to `/data/twelo/comp/`.

#     Returns:
#         None
#     """
#     remote_path = "data/comp/"
#     local_path_full = "data/twelo/comp/"

#     client = storage.Client()
#     blobs = client.list_blobs(BUCKET_NAME, prefix=remote_path, delimiter="/")

#     if os.path.exists(local_path_full):
#         print("Dataset already exists.")
#     else:
#         os.makedirs(local_path_full)
#         print(f"Downloading dataset to {local_path_full}.")
#         for blob in tqdm(blobs):
#             file_name = Path(blob.name).name
#             blob.download_to_filename(local_path_full + file_name)
#         print("Finished download.")

#     return None

def parse_original_annotations(txt_file):
    with open(txt_file) as file:
        lines = file.readlines()

    boxes = []

    for line in lines:
        line = line.split()

        x_min = float(line[1])
        y_min = float(line[2])
        x_max = float(line[3])
        y_max = float(line[4])

        boxes.append([x_min, y_min, x_max, y_max])

    columns = ['x_min', 'y_min', 'x_max', 'y_max']
    annotations = pd.DataFrame(data= boxes, columns=columns)
    print(annotations)
    return annotations

def parse_annotation(txt_file, folder_path):
    with open(txt_file) as file:
        lines = file.readlines()
        file_name = Path(file.name).stem
    file_ending = ".JPG" if PROVIDER == "" else ".jpg"
    image_path = os.path.join(folder_path, file_name + file_ending)
    boxes = []
    class_ids = []
    for line in lines:
        line = line.split()

        cls = float(line[0])
        class_ids.append(cls)

        x_min = float(line[1])
        y_min = float(line[2])
        x_max = float(line[3])
        y_max = float(line[4])

        boxes.append([x_min, y_min, x_max, y_max])

    return image_path, boxes, class_ids

def prepare_dataset(path:str, small=0):
    print("Preparing dataset.")
    txt_files = sorted(
        [
            os.path.join(path, file_name)
            for file_name in os.listdir(path)
            if file_name.endswith(".txt")
        ]
    )

    image_paths = []
    bbox = []
    classes = []
    for txt_file in txt_files:
        image_path, boxes, class_ids = parse_annotation(txt_file, path)
        if len(class_ids) != 0:
            image_paths.append(image_path)
            bbox.append(boxes)
            classes.append(class_ids)
    if small > 0:
        image_paths = image_paths[:small]
        bbox = bbox[:small]
        classes = classes[:small]

    bbox = tf.ragged.constant(bbox)
    classes = tf.ragged.constant(classes)
    image_paths = tf.ragged.constant(image_paths)

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, classes, bbox))

    return dataset

def load_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    return image

def load_dataset(image_path, classes, bbox):
    print("Loading dataset.")
    # Read Image
    def load_norm_image(image_path):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = image/255
        return image
    image = load_norm_image(image_path)
    bounding_boxes = {
        "classes": tf.cast(classes, dtype=tf.float32),
        "boxes": bbox,
    }
    return {"images": tf.cast(image, dtype=tf.float32),
            "bounding_boxes": bounding_boxes}
