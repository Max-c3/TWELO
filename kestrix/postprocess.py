import tensorflow as tf
import cv2
import numpy as np
import os
import pandas as pd

# restitch


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

    output_folder = '../data/output'

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
    #cv2.imshow('Image with Blurred Regions', image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    # save image with blurring
    output_image_path = os.path.join(output_folder, 'output_image.jpg')
    cv2.imwrite(output_image_path, image)
    print(f'Image saved to {output_image_path}')



blur_bounding_boxes(image_path, bounding_boxes)
