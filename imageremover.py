from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware  # Import CORS middleware
import numpy as np
import cv2
from rembg import remove
from io import BytesIO

app = FastAPI()

# Define CORS policies
origins = [
    "http://localhost",
    "http://localhost:5173", 
     "http://localhost:8000",
      "http://localhost:8080", # Add your frontend URL here
    # Add other allowed origins if needed
]

# Add CORS middleware to your FastAPI app
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)

async def remove_background(image):
    try:
        # Remove the background using rembg library
        output_image = remove(image)

        return output_image
    except Exception as e:
        raise e

@app.post("/remove_background/")
async def process_image(file: UploadFile = File(...)):
    try:
        # Read the uploaded image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Remove the background from the uploaded image
        output_image = await remove_background(image)
 
        # Convert the resulting image to bytes
        _, img_encoded = cv2.imencode('.png', output_image)

        # Return the resulting image
        return StreamingResponse(BytesIO(img_encoded.tobytes()), media_type="image/png")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
