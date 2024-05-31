import tensorflow as tf
import cv2
import numpy as np
import os
import pandas as pd
from kestrix.prepro_images import luc_coordinates, slicing_dictionary

# Function to 'Restitch'

def re_to_absolute_coordinates_xyxy(pred_dict):
    '''
    Input:
    pred_dict = output by the prediction of the model (dictionary with one key
    'boxes')
    Calculates the absolute coordinates of a bounding box
    for a single compartment in the context of the output
    image of size width = 4000, height = 3000
    Output: absolute coordinates in form (xyxy = x_min, y_min, x_max, y_max)
    Those need to be blurred
    '''

    # Calling the function 'luc_coordinates' and saving the resulting dictionary of coordinates in a variable
    coordinates_dict = luc_coordinates()

    # Calling the function 'slicing_dict' and saving the resulting dictionary in a variable
    slicing_dict = slicing_dictionary(coordinates_dict)

    # Transforming the output by the prediction of the model to dataframes =
    # usable format to work with afterwards
    dict_of_dfs = {}

    for i in range (0, 48):
        columns = ['x_min', 'y_min', 'x_max', 'y_max']
        data_frame = pd.DataFrame(data= pred_dict['boxes'][i], columns=columns)
        dict_of_dfs[i] = data_frame

    # Defining empty dataframe with the right column-names
    new_bounding_boxes = pd.DataFrame(columns=['x_min', 'y_min', 'x_max', 'y_max'])

    for key, value in dict_of_dfs.items():
        # key represents the number of the compartment
        # value represents the Dataframe with bounding box coordinates

        for num in range(0, value.shape[0]):

            abs_x_min = (value.iloc[num][0] + slicing_dict[key][0]) - 70
            abs_y_min = (value.iloc[num][1] + slicing_dict[key][2]) - 70
            abs_x_max = (value.iloc[num][2] + slicing_dict[key][0]) - 70
            abs_y_max = (value.iloc[num][3] + slicing_dict[key][2]) - 70

            new_bounding_boxes.loc[len(new_bounding_boxes)] = abs_x_min, abs_y_min, abs_x_max, abs_y_max

    return new_bounding_boxes       # Puts out one big Dataframe with all bounding boxes
                                    # to be blurred


# blur

file_path = '/Users/tatianalupashina/code/lupatat/temp_folder/test_input/DJI_20230504183055_0150_V.txt'
image_path = '/Users/tatianalupashina/code/lupatat/temp_folder/test_input/DJI_20230504183055_0150_V.JPG'
image = cv2.imread(image_path)

column_names = ["class", "xmin", "ymin", "xmax", "ymax"]

# Read the file with a space delimiter and assign the column names
df = pd.read_csv(file_path, names=column_names, delimiter=' ')
print(df)

# Extract bounding box coordinates and convert them to a list of tuples
bounding_boxes = df[['xmin', 'ymin', 'xmax', 'ymax']].values.tolist()
bounding_boxes

def blur_bounding_boxes(image_path, bounding_boxes):
    # Read the image
    image = cv2.imread(image_path)

    for (xmin, ymin, xmax, ymax) in bounding_boxes:
        # Check if the bounding box coordinates are within the image dimensions
        if ymin < 0 or ymax > image.shape[0] or xmin < 0 or xmax > image.shape[1]:
            print(f"Error: Bounding box coordinates ({xmin}, {ymin}, {xmax}, {ymax}) are outside the image dimensions.")
            continue

        # Extract the region of interest
        region_of_interest = image[ymin:ymax, xmin:xmax]

        # Check if the region of interest is empty
        if not region_of_interest.any():
            print(f"Error: Region of interest ({xmin}, {ymin}, {xmax}, {ymax}) is empty.")
            continue

        # Apply Gaussian blur to the region of interest with a large kernel size for more blur
        blurred_region = cv2.GaussianBlur(region_of_interest, (401, 401), 0)

        # Replace the original region with the blurred region
        image[ymin:ymax, xmin:xmax] = blurred_region

    # Display the image with the blurred regions
    cv2.imshow('Image with Blurred Regions', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



blur_bounding_boxes(image_path, bounding_boxes)
