import numpy as np
import pandas as pd
import os
from twelo.preprocess import luc_coordinates, slicing_dictionary

# Functions

def paths_and_names_files(file_type, input_path):
    '''
    Input: file_type = '.txt' or '.JPG' (string)
    input_path = path to where all our (519) training images and their label-txt.files lie
    Getting each individual path and name of the file-type.
    '''

    # Get a list of all files in the directory
    all_files = os.listdir(input_path)

    # Filter the list to include only .jpg files and sort them
    spec_files = sorted([f for f in all_files if f.endswith(file_type)])

    # Filter the list to include only the specified file types and create a dictionary
    # length = 519
    spec_files_dict = {f: os.path.join(input_path, f) for f in spec_files}


    list_files = []          # path to every single file
    list_file_names = []     # name of every single file
    a = 0

    for key, value in spec_files_dict.items():
    # print(key, value)
        if a < 525:
            list_files.append(value)
            list_file_names.append(key)
        a += 1

    return list_files, list_file_names


def getting_bounding_boxes(df):
    '''
    Requirement: Having access to the annotations for one image in
    a pd.DataFrame
    Inputs:
    df = DataFrame of one compartments' bounding boxes
    Outputs:
    image_bb = Dictionary with Key: enumerated bounding box Value: list of
    the 5 values of each bounding box (x_min, y_min, x_max, y_max, class_object)
    '''

    width_img = 4000
    height_img = 3000

    # Getting a dictionary with key: enumerated bounding box and value: list of the 5 values
    # of each bounding box (x_min, y_min, x_max, y_max, class_object)
    image_bb = {}

    for bb in range(0, df.shape[0]):

        # getting the x_center, y_center, width, height from the first bounding box
        x_center = df.iloc[bb]['x_center']                  # = 0.785663
        y_center = df.iloc[bb]['y_center']                  # = 0.776272
        first_width = df.iloc[bb]['width']                  # = 0.05003
        first_height = df.iloc[bb]['height']                # = 0.109817
        class_object = int(df.iloc[bb]['class'])            # = 0 ('car')

        #MAYBE splitting what's above and creating a for-loop for all the annotations

        # Transferring to the actual width and height of the image
        x_center = round(x_center * width_img)            # = 3143
        y_center = round(y_center * height_img)           # = 2329
        first_width = round(first_width * width_img)        # = 200
        first_height = round(first_height * height_img)     # = 329

        # Calculating the 4 cornerpoints, for which we need 4 numbers: x_min, x_max, y_min, y_max
        # (Adding 70 for each because of the padding of zeros around the original image)
        x_min = int(x_center - (first_width / 2)) + 70  # left = 3043
        x_max = int(x_center + (first_width / 2)) + 70 # right = 3243
        y_min = int(y_center - (first_height / 2))+ 70 # top = 2164
        y_max = int(y_center + (first_height / 2))+ 70 # bottom = 2493

        image_bb[bb] = [x_min, y_min, x_max, y_max, class_object]

    return image_bb



def preparing_for_loops():
    '''
    Getting an empty DataFrame with the right column-names
    and a Dictionary that we'll fill in the pipeline
    '''
    # Create an empty DataFrame with the specified columns
    columns = ['class', 'x_min', 'y_min', 'x_max', 'y_max']
    data_frame = pd.DataFrame(columns=columns)
    data_frame

    # Running this once, so that I have a dictionary full of names for the dataframes that come out of my loops
    dict_df_comps = {}

    for i in range (0, 48):
        dict_df_comps[i] = f'df_comp_{i}'

    return data_frame, dict_df_comps



def checking_bounding_box(slicing_dict, compartment_num, x_min, y_min, x_max, y_max):
    '''
    Input to this function is the number of the compartment for which it
    should be tested whether the bounding box is inside or not
    And the output of function: getting_bounding_boxes
    With this we want to take one row of the txt.file (pd.DataFrame)
    that represents one bounding box in the original image
    Output of the function:
    True if some part of the bounding box lies in the compartment
    False if the bounding box doesn't lie in the compartment!
    '''

    # If one of x_min, x_max AND one of y_min, y_max in the compartment -> True else False
    if ((x_min in range(slicing_dict[compartment_num][0], slicing_dict[compartment_num][1] + 1)
        or x_max in range(slicing_dict[compartment_num][0], slicing_dict[compartment_num][1] +1))
        and
        ((y_min in range(slicing_dict[compartment_num][2], slicing_dict[compartment_num][3] + 1)
        or y_max in range(slicing_dict[compartment_num][2], slicing_dict[compartment_num][3] + 1)))):
        tmp_var = True
    else:
        tmp_var = False

    return tmp_var



def checking_for_9_cases(slicing_dict, compartment_num, x_min, y_min, x_max, y_max):
    '''
    ONLY ACTIVATE IF function: checking_bounding_box == TRUE
    Because then we know: Some part of that bounding box lies in
    this compartment!
    '''

    # Checking for the next cases (9!)

    # Setting x_min to its value if it is in the range - otherwise setting it to be at the edge (x_min = 0!)
    if x_min in range(slicing_dict[compartment_num][0], slicing_dict[compartment_num][1] + 1):
        x_min_tmp = x_min
    else:
        x_min_tmp = slicing_dict[compartment_num][0]

    # Setting x_max to its value if it is in the range - otherwise setting it to be at the edge (x_max = 1!)
    if x_max in range(slicing_dict[compartment_num][0], slicing_dict[compartment_num][1] + 1):
        x_max_tmp = x_max
    else:
        x_max_tmp = slicing_dict[compartment_num][1]

    # Doing the same for y_min and y_max!
    if y_min in range(slicing_dict[compartment_num][2], slicing_dict[compartment_num][3] + 1):
        y_min_tmp = y_min
    else:
        y_min_tmp = slicing_dict[compartment_num][2]

    if y_max in range(slicing_dict[compartment_num][2], slicing_dict[compartment_num][3] + 1):
        y_max_tmp = y_max
    else:
        y_max_tmp = slicing_dict[compartment_num][3]

    # Returning the 4 coordinates in the XYXY class - format
    return x_min_tmp, y_min_tmp, x_max_tmp, y_max_tmp



def to_absolute_coordinates_xyxy(slicing_dict, compartment_num, x_min_tmp, y_min_tmp, x_max_tmp, y_max_tmp):
    '''
    Calculates the absolute coordinates of a bounding box
    for a single compartmecnt
    Output: absolute coordinates in form (xyxy = x_min, y_min, x_max, y_max)
    '''
    abs_x_min = x_min_tmp - slicing_dict[compartment_num][0]
    abs_x_max = x_max_tmp - slicing_dict[compartment_num][0]
    abs_y_min = y_min_tmp - slicing_dict[compartment_num][2]
    abs_y_max = y_max_tmp - slicing_dict[compartment_num][2]

    return abs_x_min, abs_y_min, abs_x_max, abs_y_max


# Function




# Now putting all of the above together to a pipeline!

# Example paths
input_path= '/Users/myself/Pictures/twelo Bootcamp Project/Training_Data_twelo/obj_Train_data'
output_path = '/Users/myself/Pictures/twelo Bootcamp Project/testing_saving_txts'


def exporting_training_txts(input_path, output_path):
    file_type_txt = '.txt'
    file_type_img = '.JPG'

    list_txts, _ = paths_and_names_files(file_type_txt, input_path)
    _, list_img_names = paths_and_names_files(file_type_img, input_path)

    for ele in range(0, len(list_txts)):

        path_annotations = list_txts[ele]

        # Read the file with a space delimiter and assign the column names
        column_names = ["class", "x_center", "y_center", "width", "height"]
        df = pd.read_csv(path_annotations, names=column_names, delimiter=' ')

        # Running necessary functions
        coordinates_dict = luc_coordinates()

        slicing_dict = slicing_dictionary(coordinates_dict)

        image_bb = getting_bounding_boxes(df)

        data_frame, dict_df_comps = preparing_for_loops()

        # For Loop for each compartment (compartment-level)
        for key, _ in slicing_dict.items():
            compartment_num = key
            dict_df_comps[key] = data_frame.copy()

            # For Loop for each bounding box
            for _, v in image_bb.items():
                x_min = v[0]
                y_min = v[1]
                x_max = v[2]
                y_max = v[3]
                class_object = v[4]

                # calling function: checking_bounding_box (Output = True or False)
                if checking_bounding_box(slicing_dict, compartment_num, x_min, y_min, x_max, y_max):

                    #calling the function: checking_for_9_cases
                    x_min_tmp, y_min_tmp, x_max_tmp, y_max_tmp = checking_for_9_cases(slicing_dict, compartment_num, x_min, y_min, x_max, y_max)

                    #calling the function: to_absolute_coordinates_xyxy
                    abs_x_min, abs_y_min, abs_x_max, abs_y_max = to_absolute_coordinates_xyxy(slicing_dict, compartment_num, x_min_tmp, y_min_tmp, x_max_tmp, y_max_tmp)

                    #Adding those to a dataframe with the original class
                    list_ready = [class_object, abs_x_min, abs_y_min, abs_x_max, abs_y_max]
                    dict_df_comps[key].loc[len(dict_df_comps[key])] = list_ready

        for comp_num, _ in dict_df_comps.items():
            # Save DataFrame to .txt file
            dict_df_comps[comp_num].to_csv(f'{output_path}/{list_img_names[ele]}_{comp_num}.txt', sep=' ',index=False, index_label = False, header=False)

        #Output = creating and exporting one .txt-file per image-compartment into a specified folder


# If all this doesn't work it can be also executed in a Jupyter Notebook 'Max_c3_execution_of_getting_txts.ipynb'
