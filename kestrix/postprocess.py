import cv2
import os
from pathlib import Path
import pandas as pd
from kestrix.preprocess import luc_coordinates, slicing_dictionary
from kestrix.params import *

def convert_coordinates_to_full_image(pred_dict):
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
        non_errors = ((data_frame["x_min"] != -1.) & (data_frame["x_max"] != -1.) & (data_frame["y_min"] != -1.) & (data_frame["y_max"] != -1.))
        data_frame[non_errors]
        dict_of_dfs[i] = data_frame


    # Cut all the lines where the row is [-1.0, -1.0, -1.0, -1.0]
    non_errors = ((dict_of_dfs["x_min"] != -1) & (dict_of_dfs["x_max"] != -1) & (dict_of_dfs["y_min"] != -1) & (dict_of_dfs["y_max"] != -1))

    dict_of_dfs[non_errors]

    # Defining empty dataframe with the right column-names
    new_bounding_boxes = pd.DataFrame(columns=['x_min', 'y_min', 'x_max', 'y_max'])

    for key, value in dict_of_dfs.items():
        # key represents the number of the compartment
        # value represents the Dataframe with bounding box coordinates

        for num in range(0, value.shape[0]):

            x_min, y_min, x_max, y_max = value.iloc[num][0], value.iloc[num][1], value.iloc[num][2], value.iloc[num][3]

            x_min = max(x_min, 0)
            x_min = min(x_min, 640)

            y_min = max(y_min, 0)
            y_min = min(y_min, 640)

            x_max = min(x_max, 640)
            x_max = max(x_max, 0)

            y_max = min(y_max, 640)
            y_max = max(y_max, 0)

            abs_x_min = (x_min + slicing_dict[key][0]) - 70
            abs_y_min = (y_min + slicing_dict[key][2]) - 70
            abs_x_max = (x_max + slicing_dict[key][0]) - 70
            abs_y_max = (y_max + slicing_dict[key][2]) - 70

            abs_x_min = max(abs_x_min, 0)
            abs_y_min = max(abs_y_min, 0)
            abs_x_max = max(abs_x_max, 0)
            abs_y_max = max(abs_y_max, 0)

            new_bounding_boxes.loc[len(new_bounding_boxes)] = int(abs_x_min), int(abs_y_min), int(abs_x_max), int(abs_y_max)

    return new_bounding_boxes       # Puts out one big Dataframe with all bounding boxes
                                    # to be blurred


def blur_bounding_boxes(image_path, new_bounding_boxes):
    output_folder = 'data/output/'
    image_name = Path(image_path).stem

    # Read the image
    image = cv2.imread(image_path)

    bounding_boxes = new_bounding_boxes[['x_min', 'y_min', 'x_max', 'y_max']].to_numpy().tolist()


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

    #Display the image with the blurred regions
    # cv2.imshow('Image with Blurred Regions', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # save image with blurring
    output_image_path = os.path.join(output_folder, f"{image_name}_blurred.jpg")
    cv2.imwrite(output_image_path, image)
    print(f'Image saved to {output_image_path}')

    return None
