import os
from pathlib import Path
from kestrix.params import *
from kestrix.registry import load_model
from kestrix.preprocess import preprocess_new_image, preprocess_training_data, preprocess_test_data
from kestrix.data import prepare_dataset, load_dataset
from kestrix.visualization import visualize_detections
import tensorflow as tf
from tensorflow import keras
import keras_cv
import matplotlib.pyplot as plt

def create_new_model():
    print("Creating new yolo model.")
     # We will use yolov8 small backbone with coco weights
    backbone = keras_cv.models.YOLOV8Backbone.from_preset(
        "yolo_v8_xs_backbone_coco",
        # increase size step by step from
        # yolo_v8_s_backbone_coco,
        # yolo_v8_m_backbone_coco,
        # yolo_v8_l_backbone_coco,
        # yolo_v8_xl_backbone_coco
        load_weights=True
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

def compile_retina_model(model):
    print("Compiling model.")
    optimizer = tf.keras.optimizers.legacy.Adam(
        learning_rate=LEARNING_RATE,
        global_clipnorm=GLOBAL_CLIPNORM,
    )
    model.compile(
        optimizer=optimizer, classification_loss="Focal", box_loss="SmoothL1"
    )

    return model

def train_model(small=0):
    model = create_new_model()
    model = compile_model(model)

    train_ds, val_ds = preprocess_training_data(small)

    callbacks = [
        keras_cv.callbacks.PyCOCOCallback(
                    val_ds,
                    BOUNDING_BOX_FORMAT
                ),
                keras.callbacks.EarlyStopping(patience=10),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.1,
                    patience=5
                )
                ]

    print("Training model.")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCH,
        callbacks=[callbacks],
    )

    return history, model

def predict(image_path, model=None):
    preprocessed_image = preprocess_new_image(image_path)

    y_pred = model.predict(preprocessed_image)

    return y_pred

def test_model(model_name):
    model = load_model(model_name)
    if model_name.startswith("Retina"):
        model = compile_retina_model(model)
    else:
        model = compile_model(model)

    test_ds = preprocess_test_data()

    results = model.evaluate(test_ds)
    visualize_detections(model, test_ds)
    #write_metrics(model_name, results)

def write_metrics(model_name, results, write_mode="a"):
    metrics = ["loss", "box_loss", "class_loss"]
    with open('metrics.txt', write_mode) as f:
        f.write(f"Model: {model_name}\n")
        for value, metric in zip(results, metrics):
            f.write(f"{metric}: ")
            f.write(f"{round(value, 2)}")
            f.write("\n")
        f.write("\n\n")

def test_all_models():
    test_ds = preprocess_test_data()

    model_list = sorted(
            [
                Path(file_name).stem
                for file_name in os.listdir("models")
                if file_name.endswith(".keras")
            ]
        )
    # with open('metrics.txt', "w") as f:
    #     f.write("")
    for model_name in model_list:
        model = load_model(model_name)
        if model_name.startswith("Retina"):
            model = compile_retina_model(model)
        else:
            model = compile_model(model)
        results = model.evaluate(test_ds)
        print("vis now")
        visualize_detections(model, test_ds, path=f"data/visualisations/{model_name}.png")
        #write_metrics(model_name, results, write_mode="a")
