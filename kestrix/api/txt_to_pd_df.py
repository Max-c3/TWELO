import pandas as pd
from keras_cv import bounding_box
import tensorflow as tf

def parse_original_annotations(txt_file):
    with open(txt_file) as file:
        lines = file.readlines()

    boxes = []

    for line in lines:
        line = line.split()

        x_min = float(line[1]) *4000
        y_min = float(line[2]) *3000
        x_max = float(line[3]) *4000
        y_max = float(line[4]) *3000

        box = tf.constant([x_min, y_min, x_max, y_max])
        converted_bounding_box = bounding_box.convert_format(box, source="CENTER_XYWH", target="XYXY")

        boxes.append(converted_bounding_box.numpy().tolist())

    columns = ['x_min', 'y_min', 'x_max', 'y_max']
    annotations = pd.DataFrame(data= boxes, columns=columns)
    annotations[columns] = annotations[columns].astype(int)

    print(annotations)
    return annotations
