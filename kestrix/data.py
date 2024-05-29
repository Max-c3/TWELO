from google.cloud import storage
import os

BUCKET_NAME = "kestrix"

def download_data():
    """
    Download the raw data from google cloud and store it under /data/raw
    """
    if os.path.exists("../data/raw"):
        print("Local data already exists. Skipping download.")
        return None
    else:
        print("Downloading data from Google Cloud")
        storage_filename = "data"
        local_filename = "data/raw"

        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(storage_filename)
        blob.download_to_filename(local_filename)
        return None
