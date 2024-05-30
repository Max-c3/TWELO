from google.cloud import storage
import os
from pathlib import Path
import tensorflow as tf

BUCKET_NAME = "kestrix"

def download_raw_data():
    """Download raw data from google cloud to `../data/kestrix/full/raw/`.

    Returns:
        None
    """
    remote_path = "data/raw/"
    local_path_full = "../data/kestrix/full/raw/"

    client = storage.Client()
    blobs = client.list_blobs(BUCKET_NAME, prefix=remote_path, delimiter="/")

    if os.path.exists(local_path_full):
        print("Dataset already exists.")
    else:
        os.makedirs(local_path_full)
        print(f"Downloading dataset to {local_path_full}.")
        for blob in blobs:
            file_name = Path(blob.name).name
            blob.download_to_filename(local_path_full + file_name)
        print("Finished download.")

    return None

def convert_image_to_tensor(image_path:str) -> tf.Tensor:
    """Convert an image to a tensor

    Args:
        image_path (str): File path of the image.

    Returns:
        tf.Tensor: Image decoded as a tensor.
    """
    if not os.path.exists(image_path):
        print(f"File not found: {image_path}")
        return None
    else:
        # Read the file contents as a string tensor
        image_string = tf.io.read_file(image_path)
        # Decode the JPEG image to a uint8 tensor
        decoded_image = tf.image.decode_jpeg(image_string, channels=3)

        return decoded_image

