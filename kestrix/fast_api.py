from fastapi import FastAPI, File, UploadFile, Response
import os
from PIL import Image
from io import BytesIO
from tensorflow import keras

from kestrix.model import load_model, predict, compile_model
from kestrix.postprocess import convert_coordinates_to_full_image, blur_bounding_boxes

app = FastAPI()

loaded_model =  keras.saving.load_model("/Users/foxidy/code/Max-c3/Kestrix_Project/models/compartment_20240531.keras")
model = compile_model(loaded_model)

@app.get("/")
def root():
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
        ## --> Input correct input_image location
        input_folder = 'data/input'
        input_image_path = os.path.join(input_folder, f"{file.filename}")

        input_image.save(input_image_path, format='JPEG')

        output_folder = 'data/output'
        output_file_name = os.path.splitext(os.path.basename(file.filename))[0]
        output_image_path = os.path.join(output_folder, f"{output_file_name}_blurred.jpg")
        print(output_image_path)

        ## Pipeline to apply Model predict, stitch batch bounding boxes coordinates and blur image
        # Apply Model predict
        compartments_bounding_boxes = predict(input_image_path, model)

        # Stitch back bounding boxes of the compartments to image
        image_bounding_boxes = convert_coordinates_to_full_image(compartments_bounding_boxes)

        # Blur input image based of image bounding boxes
        blur_bounding_boxes(input_image_path, image_bounding_boxes)

        # Retrieve blurred image
        output_folder = '/Users/foxidy/code/Max-c3/Kestrix_Project/data/output'
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
