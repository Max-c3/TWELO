# Imports needed

import numpy as np
from PIL import Image
from kestrix.data import pad_image

# import matplotlib.pyplot as plt

# Function (1) - needed to be called by Function (2)

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

# Function (2)

def splitting_into_compartments(tensor, output_path):
    '''
    Takes as an input the Tensor that represents the
    whole image (+ padding on each side of 70 pixels already added)
    that is supposed to be split into compartments.
    The tensor should thus have the shape (3140, 4140, 3)
    '''

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

    # Saving each compartment to the output_path (here: /data/kestrix/temp) as a
    for i in range(0, 48):
        compartment = compartment_tensors[i,:,:,:]  # object shape (640, 640, 3)
        image = Image.fromarray(compartment)
        image.save(f'{output_path}comp_{i}.jpg')

    return compartment_tensors     # 48 Tensors of shape (640, 640, 3)
