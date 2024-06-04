import os
from pathlib import Path
from cloud_detect import async_provider

PROVIDER = async_provider()

BOUNDING_BOX_FORMAT = "xyxy"
CLASS_MAPPING = {
    0: "car",
    1: "person"
}

SPLIT_RATIO = 0.2
BATCH_SIZE = 4
LEARNING_RATE = 0.001
EPOCH = 2
GLOBAL_CLIPNORM = 10.0

BUCKET_NAME = "kestrix"
