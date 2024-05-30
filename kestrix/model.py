from kestrix.params import *
from kestrix.data import prepare_dataset, load_dataset, load_image, pad_image
from kestrix.registry import save_model, load_model
import tensorflow as tf
from tensorflow.keras import keras
import keras_cv

def compile_model(model):
    optimizer = tf.keras.optimizers.legacy.Adam(
        learning_rate=LEARNING_RATE,
        global_clipnorm=GLOBAL_CLIPNORM,
    )
    model = model.compile(
        optimizer=optimizer, classification_loss="binary_crossentropy", box_loss="ciou"
    )

    return model

def train()->history:
    model = load_model()

    train_ds, val_ds = preprocess_data()

    coco_metrics_callback = keras_cv.callbacks.PyCOCOCallback(
        val_ds,
        bounding_box_format)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=2,
        callbacks=[coco_metrics_callback],
    )
    save_model(model)

    return history, model

def predict(input, model=None):
    if not model:
        model = load_model("../models/TODO:")

    preprocessed_image = preprocess_new_image(input)


    y_pred = model.predict(preprocessed_image)

    return y_pred

def preprocess_new_image(path):
    padded_image = pad_image(path)

    splitted_images #TODO receive as tensor with first dimension different images
    splitted_images_float = tf.cast(splitted_images, dtype=tf.float32)
    new_data = tf.data.Dataset.from_tensor_slices(splitted_images)
    new_data = new_data.prefetch(num_parallel_calls=tf.data.AUTOTUNE)

    return new_data


def preprocess_data() -> (train_ds, val_ds):
    path = "../data/kestrix/comp"

    def dict_to_tuple(inputs):
        return inputs["images"], bounding_box.to_dense(
            inputs["bounding_boxes"], max_boxes=32
        )

    data = prepare_dataset(path)

    # Splitting data
    # Determine number of validation data
    num_val = int(len(txt_files) * SPLIT_RATIO)

    # split into train and validation
    # TODO change into random split via train_test_split
    val_data = data.take(num_val)
    train_data = data.skip(num_val)

    # Image Augmentation: https://keras.io/api/keras_cv/layers/augmentation/
    augmenter = keras.Sequential(
        layers=[
            keras_cv.layers.RandomFlip(
                mode="horizontal",
                bounding_box_format=bounding_box_format),
            keras_cv.layers.RandomShear(
                x_factor=0.2,
                y_factor=0.2,
                bounding_box_format=bounding_box_format
            )
        ]
    )

    resizing = keras_cv.layers.JitteredResize(
        target_size=(640, 640),
        scale_factor=(0.8, 1),
        bounding_box_format=bounding_box_format,
    )

    # create training dataset
    train_ds = train_data.map(load_dataset, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.shuffle(BATCH_SIZE * 4)
    train_ds = train_ds.ragged_batch(BATCH_SIZE, drop_remainder=True)
    train_ds = train_ds.map(augmenter, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.map(resizing, num_parallel_calls=tf.data.AUTOTUNE)

    # create validation dataset
    val_ds = val_data.map(load_dataset, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.shuffle(BATCH_SIZE * 4)
    val_ds = val_ds.ragged_batch(BATCH_SIZE, drop_remainder=True)
    val_ds = val_ds.map(resizing, num_parallel_calls=tf.data.AUTOTUNE)

    train_ds = train_ds.map(dict_to_tuple, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

    val_ds = val_ds.map(dict_to_tuple, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds
