import os
from pathlib import Path
from cloud_detect import provider

PROVIDER = provider()

BOUNDING_BOX_FORMAT = "xyxy"
CLASS_MAPPING = {
    "car": 0,
    "person": 1,
    "empty": 2
}

SPLIT_RATIO = 0.2
BATCH_SIZE = 4
LEARNING_RATE = 0.001
EPOCH = 1
GLOBAL_CLIPNORM = 10.0

BUCKET_NAME = "kestrix"
