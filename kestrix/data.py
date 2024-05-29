from google.cloud import storage
import os
from pathlib import Path

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
