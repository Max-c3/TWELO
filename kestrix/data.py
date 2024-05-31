from google.cloud import storage
import os
from pathlib import Path
import tensorflow as tf
from tqdm import tqdm

BUCKET_NAME = "kestrix"

def download_raw_data():
    """Download raw data from google cloud to `../data/kestrix/raw/`.

    Returns:
        None
    """
    remote_path = "data/raw/"
    local_path_full = "../data/kestrix/raw/"

    client = storage.Client()
    blobs = client.list_blobs(BUCKET_NAME, prefix=remote_path, delimiter="/")

    if os.path.exists(local_path_full):
        print("Dataset already exists.")
    else:
        os.makedirs(local_path_full)
        print(f"Downloading dataset to {local_path_full}.")
        for blob in tqdm(blobs):
            file_name = Path(blob.name).name
            blob.download_to_filename(local_path_full + file_name)
        print("Finished download.")

    return None

def download_comp_data():
    """Download compartmented data from google cloud to `../data/kestrix/comp/`.

    Returns:
        None
    """
    remote_path = "data/comp/"
    local_path_full = "../data/kestrix/comp/"

    client = storage.Client()
    blobs = client.list_blobs(BUCKET_NAME, prefix=remote_path, delimiter="/")

    if os.path.exists(local_path_full):
        print("Dataset already exists.")
    else:
        os.makedirs(local_path_full)
        print(f"Downloading dataset to {local_path_full}.")
        for blob in tqdm(blobs):
            file_name = Path(blob.name).name
            blob.download_to_filename(local_path_full + file_name)
        print("Finished download.")

    return None

def convert_image_to_tensor(image_path:str) -> tf.Tensor:
    """Convert an image to a tensor

    Args:
        image_path (str): File path of the image.

    Returns:
        tf.Tensor: Image decoded as a tensor.
    """
    if not os.path.exists(image_path):
        print(f"File not found: {image_path}")
        return None
    else:
        # Read the file contents as a string tensor
        image = tf.io.read_file(image_path)
        # Decode the JPEG image to a uint8 tensor
        decoded_image = tf.image.decode_jpeg(image, channels=3)

        return decoded_image

def pad_image(input_path:str, padding_amount:int=70) -> tf.Tensor:
    """Returns the image padded as a Tensor.

    Parameters
    ----------
    input_path : str
        Input path.
    padding_amount : int, optional
        Amount of padding at each side, by default 70

    Returns
    -------
    tf.Tensor
        The Tensor of the padded image
    """
    decoded_image = convert_image_to_tensor(input_path)

    # Define paddings
    paddings = tf.constant([[padding_amount, padding_amount], [padding_amount, padding_amount]])  # for height and width

    # Initialize an empty list to store padded channels
    padded_channels = []

    # Loop through each channel and apply padding
    for i in range(decoded_image.shape[2]):  # Loop through the 3 channels
        channel = decoded_image[:, :, i]  # Extract the i-th channel
        padded_channel = tf.pad(channel, paddings, "CONSTANT")  # Apply padding
        padded_channels.append(padded_channel)  # Add to list

    # Stack the padded channels back together
    padded_image = tf.stack(padded_channels, axis=2)
    assert(padded_image.shape == tf.TensorShape([3140, 4140, 3]))

    return padded_image


def pad_all_images(input_dir:str, padding_amount=70) -> tf.Tensor:
    """Pads all images at the specified directory and returns all as a tensor.

    Parameters
    ----------
    input_dir : str
        Path of the input directory.
    padding_amount : int, optional
        Amount of padding at each side, by default 70

    Returns
    -------
    tf.Tensor
        The Tensor containing all images in the directory.
    """
    print("Padding all images.")

    padded_images = []

    # Get the list of image files in the input directory
<<<<<<< HEAD
    image_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg'))])
=======
    image_files = sorted(
        [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg'))]
        )
>>>>>>> postprocessing

    for image_file in tqdm(image_files):
        image_path = os.path.join(input_dir, image_file)

        padded_image = pad_image(image_path, padding_amount=padding_amount)

        padded_images.append(padded_image)

    tf_padded_images = tf.stack(padded_images, axis=0)

    assert(tf_padded_images.shape == tf.TensorShape([len(image_files), 3140, 4140, 3]))

    print(f"Finished padding all images. Output shape: {tf_padded_images.shape.as_list()}")

    return tf_padded_images


def parse_annotation(txt_file, folder_path):
    with open(txt_file) as file:
        lines = file.readlines()
        file_name = Path(file.name).stem

    image_path = os.path.join(folder_path, file_name + ".JPG")
    boxes = []
    class_ids = []
    for line in lines:
        line = line.split()

        cls = float(line[0])
        class_ids.append(cls)

        x_min = float(line[1])
        y_min = float(line[2])
        x_max = float(line[3])
        y_max = float(line[4])

        boxes.append([x_min, y_min, x_max, y_max])

    return image_path, boxes, class_ids

def prepare_dataset(path:str):
    txt_files = sorted(
        [
            os.path.join(path, file_name)
            for file_name in os.listdir(path)
            if file_name.endswith(".txt")
        ]
    )

    image_paths = []
    bbox = []
    classes = []
    for txt_file in txt_files:
        image_path, boxes, class_ids = parse_annotation(txt_file, path)
        image_paths.append(image_path)
        bbox.append(boxes)
        classes.append(class_ids)

    bbox = tf.ragged.constant(bbox)
    classes = tf.ragged.constant(classes)
    image_paths = tf.ragged.constant(image_paths)

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, classes, bbox))

    return dataset

def load_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    return image

def load_dataset(image_path, classes, bbox):
    # Read Image
    image = load_image(image_path)
    bounding_boxes = {
        "classes": tf.cast(classes, dtype=tf.float32),
        "boxes": bbox,
    }
    return {"images": tf.cast(image, dtype=tf.float32),
            "bounding_boxes": bounding_boxes}

