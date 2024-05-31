from fastapi import FastAPI, File, UploadFile, Response
import os
from PIL import Image
from io import BytesIO

from kestrix.model import load_model, predict
from kestrix.postprocess import converting_coordinates_to_full_image, blur_bounding_boxes

app = FastAPI()

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
        input_image_path = os.path.expanduser(f"~/Downloads/processed_{file.filename}")
        input_image.save(input_image_path, format='JPEG')

        ## Pipeline to apply Model predict, stitch batch bounding boxes coordinates and blur image
        # Apply Model predict
        model = load_model("..models/")
        compartments_bounding_boxes = predict(input_image_path, model)

        # Stitch back bounding boxes of the compartments to image
        image_bounding_boxes = converting_coordinates_to_full_image(compartments_bounding_boxes)

        # Blur input image based of image bounding boxes
        blur_bounding_boxes(image_bounding_boxes)
        blurred_image_path = os.path.join(output_folder, f"{image_name}_blurred.jpeg")
        blurred_image =

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

@app.post("/upload_")
async def upload(file: UploadFile = File(...)):
    try:
        # Read the contents of the uploaded file
        contents = await file.read()

        # Convert the byte contents into an image
        image = Image.open(BytesIO(contents))

        # Rotate Image By 180 Degree
        rotated_image = image.rotate(180)

        # Save the image to a BytesIO object
        bytes_io = BytesIO()
        rotated_image.save(bytes_io, format='JPEG')
        bytes_io.seek(0)

        # Expand the user's home directory for saving the image
        file_location = os.path.expanduser(f"~/Downloads/processed_{file.filename}")
        rotated_image.save(file_location, format='JPEG')

        return Response(content=bytes_io.getvalue(), media_type="image/jpeg")
        # {"INFO": f"File '{file.filename}' uploaded to the API and saved successfully to {file_location}."}

    except Exception as e:
        # Log the error
        print(f"Error processing uploaded file: {e}")

        # Return the error message
        return {"message": f"There was an error processing the file: {str(e)}"}
