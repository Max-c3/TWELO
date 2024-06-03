from fastapi import FastAPI, File, UploadFile, Response
import os
from PIL import Image
from io import BytesIO

app = FastAPI()

@app.get("/")
def root():
    return {'API check': 'Hello, endpoint connected.'}

@app.post("/upload")
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
