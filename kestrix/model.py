from kestrix.params import *
from kestrix.registry import save_model, load_model
from kestrix.preprocess import preprocess_new_image, preprocess_data
import tensorflow as tf
from tensorflow import keras
import keras_cv

def create_new_model():
    print("Creating new yolo model.")
     # We will use yolov8 small backbone with coco weights
    backbone = keras_cv.models.YOLOV8Backbone.from_preset(
        "yolo_v8_s_backbone_coco"
    )

    prediction_decoder = keras_cv.layers.NonMaxSuppression(
        bounding_box_format=BOUNDING_BOX_FORMAT,
        from_logits=True,
        iou_threshold=0.2,
        confidence_threshold=0.7,
    )

    model = keras_cv.models.YOLOV8Detector(
        num_classes=len(CLASS_MAPPING),
        bounding_box_format=BOUNDING_BOX_FORMAT,
        backbone=backbone,
        fpn_depth=1,
        prediction_decoder=prediction_decoder
    )
    model = compile_model(model)
    return model

def compile_model(model):
    print("Compiling model.")
    optimizer = tf.keras.optimizers.legacy.Adam(
        learning_rate=LEARNING_RATE,
        global_clipnorm=GLOBAL_CLIPNORM,
    )
    model.compile(
        optimizer=optimizer, classification_loss="binary_crossentropy", box_loss="ciou"
    )

    return model

def train_model(new=True, model_path=None):
    model = create_new_model() if new else load_model(model_path)

    train_ds, val_ds = preprocess_data()

    coco_metrics_callback = keras_cv.callbacks.PyCOCOCallback(
        val_ds,
        BOUNDING_BOX_FORMAT)

    print("Training model.")
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
