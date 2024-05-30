from keras_cv import visualization
from kestrix.params import *
from kestrix.data import load_dataset, prepare_dataset
import tensorflow as tf


def visualize_bounding_box(path:str) -> None:
    """Visualize one batch of bounding boxes.

    Args:
        path (str): Path to the images/txt
    """

    prepared_dataset = prepare_dataset(path)

    loaded_dataset = prepared_dataset.map(load_dataset, num_parallel_calls=tf.data.AUTOTUNE)

    batched_dataset = loaded_dataset.ragged_batch(BATCH_SIZE, drop_remainder=True)

    inputs = next(iter(batched_dataset.take(1)))

    images, bounding_boxes = inputs["images"], inputs["bounding_boxes"]
    visualization.plot_bounding_box_gallery(
        images,
        value_range=(0, 255),
        rows=2,
        cols=2,
        y_true=bounding_boxes,
        scale=5,
        font_scale=0.7,
        bounding_box_format=BOUNDING_BOX_FORMAT,
        class_mapping=CLASS_MAPPING,
    )
