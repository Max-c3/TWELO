from fastapi import FastAPI, File, UploadFile, Response
import os
import time
from PIL import Image
from io import BytesIO

from kestrix.model import load_model, predict, compile_model, compile_retina_model
from kestrix.postprocess import convert_coordinates_to_full_image, blur_bounding_boxes
from kestrix.api.txt_to_pd_df import parse_original_annotations

app = FastAPI()

model_name = "compartment_20240531"
loaded_model =  load_model(model_name)
if model_name.startswith('Reti'):
    model = compile_retina_model(loaded_model)
else:
    model = compile_model(loaded_model)

@app.get("/")
async def root():
    # Ensure provider is initialized
    return {'API check': 'Hello, API endpoint connected.'}

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    try:
        # Read the contents of the uploaded file
        contents = await file.read()

        # Convert the byte contents into an image
        input_image = Image.open(BytesIO(contents))

        # Save the image to a BytesIO object
        bytes_io = BytesIO()
        input_image.save(bytes_io, format='JPEG')
        bytes_io.seek(0)

        # Expand the user's home directory for saving the image
        input_folder = 'data/input'
        input_image_path = os.path.join(input_folder, f"{file.filename}")

        input_image.save(input_image_path, format='JPEG')

        ## Pipeline to apply Model predict, stitch batch bounding boxes coordinates and blur image
        # Apply Model predict
        start = time.time()
        compartments_bounding_boxes = predict(input_image_path, model)
        end = time.time()
        print(f"Prediction took for {end - start}.")

        # Stitch back bounding boxes of the compartments to image
        start = time.time()
        image_bounding_boxes = convert_coordinates_to_full_image(compartments_bounding_boxes)
        end = time.time()
        print(f"Stitching back the bounding boxes took {end - start}.")
        print(image_bounding_boxes)

        # Blur input image based of image bounding boxes
        start = time.time()
        blur_bounding_boxes(input_image_path, image_bounding_boxes)
        end = time.time()
        print(f"Blurring the uploaded image for the bounding boxes took {end - start}.")

        # Retrieve blurred image
        output_folder = 'data/output'
        output_file_name = os.path.splitext(os.path.basename(file.filename))[0]
        output_image_path = os.path.join(output_folder, f"{output_file_name}_blurred.jpg")
        blurred_image = Image.open(output_image_path)

        # Save the image to a BytesIO object
        bytes_io = BytesIO()
        blurred_image.save(bytes_io, format='JPEG')
        bytes_io.seek(0)

        return Response(content=bytes_io.getvalue(), media_type="image/jpeg")
        # {"INFO": f"File '{file.filename}' uploaded to the API and saved successfully to {file_location}."}

    except Exception as e:
        # Log the error
        print(f"Error processing uploaded file: {e}")

        # Return the error message
        return {"message": f"There was an error processing the file: {str(e)}"}

@app.post("/upload_twelo25") # To work, you need 'data/bounding_boxes' repository including the .txt of the bounding boxes.
async def upload(file: UploadFile = File(...)):
    try:
        # Read the contents of the uploaded file
        start = time.time()
        contents = await file.read()

        # Convert the byte contents into an image
        input_image = Image.open(BytesIO(contents))

        # Save the image to a BytesIO object
        bytes_io = BytesIO()
        input_image.save(bytes_io, format='JPEG')
        bytes_io.seek(0)

        # Expand the user's home directory for saving the image
        input_folder = 'data/input'
        input_image_path = os.path.join(input_folder, f"{file.filename}")

        input_image.save(input_image_path, format='JPEG')

        end = time.time()
        print(f"Saving the uploaded file for processing took {end - start}.")

        ## Pipeline to identify .txt file with the bounding boxes and blur image
        # Identify .txt file with the bounding boxes
        start = time.time()
        bounding_boxes_folder = 'data/bounding_boxes'
        filename = os.path.splitext(os.path.basename(file.filename))[0]
        image_bounding_boxes_txt = os.path.join(bounding_boxes_folder, f"{filename}.txt")
        end = time.time()
        print(f"Identifying .txt file with the bounding boxes took {end - start}.")

        # Transform .txt file into Pandas Dataframe
        start = time.time()
        image_bounding_boxes = parse_original_annotations(image_bounding_boxes_txt)
        end = time.time()
        print(f"Transform .txt file into Pandas Dataframe took for {end - start}.")

        # Blur input image based of image bounding boxes
        start = time.time()
        blur_bounding_boxes(input_image_path, image_bounding_boxes)
        end = time.time()
        print(f"Blurring the uploaded image for the bounding boxes took {end - start}.")

        # Retrieve blurred image
        output_folder = 'data/output'
        output_file_name = os.path.splitext(os.path.basename(file.filename))[0]
        output_image_path = os.path.join(output_folder, f"{output_file_name}_blurred.jpg")
        blurred_image = Image.open(output_image_path)

        # Save the image to a BytesIO object
        bytes_io = BytesIO()
        blurred_image.save(bytes_io, format='JPEG')
        bytes_io.seek(0)

        return Response(content=bytes_io.getvalue(), media_type="image/jpeg")
        # {"INFO": f"File '{file.filename}' uploaded to the API and saved successfully to {file_location}."}

    except Exception as e:
        # Log the error
        print(f"Error processing uploaded file: {e}")

        # Return the error message
        return {"message": f"There was an error processing the file: {str(e)}"}
