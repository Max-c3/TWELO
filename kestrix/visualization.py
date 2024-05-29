from keras_cv import visualization
from kestrix.params import *
import tensorflow as tf

def visualize_single_bounding_box(image:tf.Tensor, bbox:dict):
    visualization.plot_bounding_box_gallery(
            image,
            value_range=(0, 255),
            rows=1,
            cols=1,
            y_pred=bbox,
            scale=5,
            font_scale=0.7,
            bounding_box_format=BOUNDING_BOX_FORMAT,
            class_mapping=CLASS_MAPPING,
        )
