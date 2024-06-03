import os
from pathlib import Path
from cloud_detect import async_provider, provider

PROVIDER = async_provider()

BOUNDING_BOX_FORMAT = "xyxy"
CLASS_MAPPING = {
    "car": 0,
    "person": 1
}

SPLIT_RATIO = 0.2
BATCH_SIZE = 2
LEARNING_RATE = 0.001
EPOCH = 2
GLOBAL_CLIPNORM = 10.0

BUCKET_NAME = "kestrix"
