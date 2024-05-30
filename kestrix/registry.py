from kestrix.params import *
import datetime

def load_model(model_path=None):
    if not Model:
        model = keras.saving.load_model(model_path)
        return model
    else:
        # We will use yolov8 small backbone with coco weights
        backbone = keras_cv.models.YOLOV8Backbone.from_preset(
            "yolo_v8_s_backbone_coco"
        )

        prediction_decoder = keras_cv.layers.NonMaxSuppression(
            bounding_box_format="xywh",
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

def save_model(model):
    now = datetime.datetime.now()
    model.save(f"model_{now}.keras")
