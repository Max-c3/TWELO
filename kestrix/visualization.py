from keras_cv import visualization
from kestrix.params import *
import tensorflow as tf

def visualize_single_bounding_box(image:tf.Tensor, bbox:dict) -> None:
    """Visualize a single bounding box.

    Args:
        image (tf.Tensor): Image as a tensor.
        bbox (dict): A dictionary of the format
                    ```
                    {
                        "classes": [classes],
                        "bounding_boxes": [lists of the coordinates of the boxes]
                    }
                    ```
    """
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
