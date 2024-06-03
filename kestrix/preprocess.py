import numpy as np
import os
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import tqdm
import keras_cv
from keras_cv import bounding_box

from kestrix.data import prepare_dataset, load_dataset
from kestrix.params import *


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

    print("Padding image.")
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
    image_files = sorted(
        [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg'))]
        )

    for image_file in tqdm(image_files):
        image_path = os.path.join(input_dir, image_file)

        padded_image = pad_image(image_path, padding_amount=padding_amount)

        padded_images.append(padded_image)

    tf_padded_images = tf.stack(padded_images, axis=0)

    assert(tf_padded_images.shape == tf.TensorShape([len(image_files), 3140, 4140, 3]))

    print(f"Finished padding all images. Output shape: {tf_padded_images.shape.as_list()}")

    return tf_padded_images


def luc_coordinates():
    ''' A function that takes as an input (given in the function already)
    width (of the image) = 4000
    height (of the image) = 3000
    width_comp = number of compartments for the width (e.g. 8)
    height_comp = number of compartments for the height (e.g. 6)
    comp_size = size of each compartment (e.g. 640)
    and returns the coordinates of the left-upper-corner (luc) of
    each compartment
    '''

    # Given inputs
    width = 4000
    height = 3000
    num_width_comp = 8
    num_height_comp = 6
    comp_size = 640

    # Step size is the size of the compartments unique pixels horizontally and vertically
    step_size_horiz = width / num_width_comp
    step_size_vert = height / num_height_comp

    # Defining how much overlap there will be between each compartment vertically and horizontally
    # overlap_horiz = int(((comp_size * num_width_comp) - width)/num_width_comp)
    # overlap_vert = int(((comp_size * num_height_comp) - height)/num_height_comp)

    #creating a list with all the x_coordinates for the compartments (8)
    x_coordinates = [0]
    [x_coordinates.append(int(ele * step_size_horiz)) for ele in range(1, num_width_comp)]

    # creating a list with all the y_coordinates for the compartments (6)
    y_coordinates = [0]
    [y_coordinates.append(int(ele * step_size_vert)) for ele in range(1, num_height_comp)]


    # Creating a dictionary with the coordinates
    coordinates_dict = {}
    b = 0 # no inherent important meaning

    for ele in range (0, num_height_comp):
        for i in range (0, num_width_comp):
            a = [x_coordinates[i], y_coordinates[ele]]
            b += 1
            coordinates_dict[f'cor_{b}'] = a

    return coordinates_dict


def slicing_dictionary(coordinates_dict):

    # Calling the function 'luc_coordinates' and saving the resulting dictionary of coordinates in a variable
    # coordinates_dict = luc_coordinates(num_width_comp, num_height_comp, comp_size)

    slize_size = 640

    # Turning the coordinates into a list
    coordinates_list = [value for key, value in coordinates_dict.items()]

    # creating a for loop for getting the correct slizing integers and putting them into a dictionary
    slicing_dict = {}

    for i in range (0, 48):
        slice_1, slice_2 = coordinates_list[i][0], coordinates_list[i][1]
        slice_1_a = slice_1
        slice_1_b = slice_1_a + slize_size
        slice_2_a = slice_2
        slice_2_b = slice_2_a + slize_size
        slicing_dict[i] = [slice_1_a, slice_1_b, slice_2_a, slice_2_b]


    return slicing_dict

def split_into_compartments(tensor:tf.Tensor, output_path=None):
    '''
    Takes as an input the Tensor that represents the
    whole image (+ padding on each side of 70 pixels already added)
    that is supposed to be split into compartments.
    The tensor should thus have the shape (3140, 4140, 3)
    '''
    assert(tensor.shape == (3140, 4140, 3))
    print("Splitting into compartments.")


    # Calling the function 'luc_coordinates' and saving the resulting dictionary of coordinates in a variable
    coordinates_dict = luc_coordinates()

    # Calling the function 'slicing_dict' and saving the resulting dictionary in a variable
    slicing_dict = slicing_dictionary(coordinates_dict)

    # slicing the input-tensor to get (48) of shape (640, 640, 3)
    #version tensor.shape = (3140, 4140, 3)
    list_of_tensors = [tensor[slicing_dict[ele][2]:slicing_dict[ele][3],
                                slicing_dict[ele][0]:slicing_dict[ele][1],

                                    ] for ele in range (0, 48)]

    # To np.array
    compartment_tensors = np.array(list_of_tensors)

    if output_path:
        # Saving each compartment to the output_path as a jpg
        for i in range(0, 48):
            compartment = compartment_tensors[i,:,:,:]  # object shape (640, 640, 3)
            image = Image.fromarray(compartment)
            image.save(f'{output_path}comp_{i}.jpg')

    return compartment_tensors      # 48 Tensors of shape (640, 640, 3)


def preprocess_new_image(path):
    """Processes single image to be fed into model

    Parameters
    ----------
    path : str
        Path to the image

    Returns
    -------
    _type_
        tensorflow Dataset
    """
    padded_image = pad_image(path)

    splitted_images = split_into_compartments(padded_image)
    splitted_images_float = tf.cast(splitted_images, dtype=tf.float32)
    splitted_images_float_batched = tf.expand_dims(splitted_images_float, axis=0)
    new_data = tf.data.Dataset.from_tensor_slices(splitted_images_float_batched)
    new_data = new_data.prefetch(tf.data.AUTOTUNE)

    return new_data


def preprocess_training_data(small=0):
    path = "data/kestrix/train"

    def dict_to_tuple(inputs):
        return inputs["images"], bounding_box.to_dense(
            inputs["bounding_boxes"], max_boxes=32
        )

    data = prepare_dataset(path, small=small)

    # Splitting data
    # Determine number of validation data
    num_val = int(len(data) * SPLIT_RATIO)

    # split into train and validation
    val_data = data.take(num_val)
    train_data = data.skip(num_val)

    # Image Augmentation: https://keras.io/api/keras_cv/layers/augmentation/
    augmenter = keras.Sequential(
        layers=[
            keras_cv.layers.RandomFlip(
                mode="horizontal",
                bounding_box_format=BOUNDING_BOX_FORMAT),
            keras_cv.layers.RandomShear(
                x_factor=0.2,
                y_factor=0.2,
                bounding_box_format=BOUNDING_BOX_FORMAT
            )
        ]
    )

    resizing = keras_cv.layers.JitteredResize(
        target_size=(640, 640),
        scale_factor=(0.8, 1),
        bounding_box_format=BOUNDING_BOX_FORMAT,
    )

    no_actual_resizing = keras_cv.layers.Resizing(
        640, 640, bounding_box_format="xyxy",
        pad_to_aspect_ratio=True
    )

    # create training dataset
    train_ds = train_data.map(load_dataset, num_parallel_calls=tf.data.AUTOTUNE)
    # train_ds = train_ds.shuffle(BATCH_SIZE * 4)
    train_ds = train_ds.ragged_batch(BATCH_SIZE, drop_remainder=True)
    train_ds = train_ds.map(augmenter, num_parallel_calls=tf.data.AUTOTUNE)
    # train_ds = train_ds.map(resizing, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.map(no_actual_resizing, num_parallel_calls=tf.data.AUTOTUNE)

    # create validation dataset
    val_ds = val_data.map(load_dataset, num_parallel_calls=tf.data.AUTOTUNE)
    # val_ds = val_ds.shuffle(BATCH_SIZE * 4)
    val_ds = val_ds.ragged_batch(BATCH_SIZE, drop_remainder=True)
    # val_ds = val_ds.map(resizing, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(no_actual_resizing, num_parallel_calls=tf.data.AUTOTUNE)


    train_ds = train_ds.map(dict_to_tuple, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

    val_ds = val_ds.map(dict_to_tuple, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds
