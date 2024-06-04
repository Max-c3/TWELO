from keras_cv import visualization, bounding_box
from kestrix.params import *
from kestrix.data import load_dataset, prepare_dataset
import tensorflow as tf
import matplotlib.pyplot as plt


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

def visualize_detections(model, dataset):
    images, y_true = next(iter(dataset.take(1)))
    images = images*255
    y_pred = model.predict(images)
    y_pred = bounding_box.to_ragged(y_pred)
    visualization.plot_bounding_box_gallery(
        images,
        value_range=(0, 255),
        bounding_box_format=BOUNDING_BOX_FORMAT,
        y_true=y_true,
        y_pred=y_pred,
        scale=4,
        rows=2,
        cols=2,
        show=True,
        font_scale=0.7,
        class_mapping=CLASS_MAPPING,
        legend=True
    )

def plot_history(history, title=None):
    fig, ax = plt.subplots(3,1, figsize=(20,7))

    # --- LOSS ---

    ax[0].plot(history.history['loss'])
    ax[0].plot(history.history['val_loss'])

    ax[0].set_title('Model loss')
    ax[0].set_ylabel('Loss')
    ax[0].set_xlabel('Epoch')

    ax[0].legend(['Train', 'Test'], loc='best')

    ax[0].grid(axis="x",linewidth=0.5)
    ax[0].grid(axis="y",linewidth=0.5)

    # --- BOX LOSS

    ax[1].plot(history.history['box_loss'])
    ax[1].plot(history.history['val_box_loss'])

    ax[1].set_title('Model Box Loss')
    ax[1].set_ylabel('Box Loss')
    ax[1].set_xlabel('Epoch')

    ax[1].legend(['Train', 'Test'], loc='best')


    ax[1].grid(axis="x",linewidth=0.5)
    ax[1].grid(axis="y",linewidth=0.5)

    # --- CLASS LOSS

    ax[2].plot(history.history['class_loss'])
    ax[2].plot(history.history['val_class_loss'])

    ax[2].set_title('Model Class Loss')
    ax[2].set_ylabel('Class Loss')
    ax[2].set_xlabel('Epoch')

    ax[2].legend(['Train', 'Test'], loc='best')

    ax[2].grid(axis="x",linewidth=0.5)
    ax[2].grid(axis="y",linewidth=0.5)


    if title:
        fig.suptitle(title)

    plt.show()
