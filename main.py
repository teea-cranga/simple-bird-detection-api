from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import torch
import io
from PIL import Image
from transformers import EfficientNetImageProcessor, EfficientNetForImageClassification

# start application
app = FastAPI()

# Loading the model and preprocessor from HuggingFace
preprocessor = EfficientNetImageProcessor.from_pretrained("dennisjooo/Birds-Classifier-EfficientNetB2")
model = EfficientNetForImageClassification.from_pretrained("dennisjooo/Birds-Classifier-EfficientNetB2")

@app.post("/predict")
async def predict_picture(file: UploadFile = File(...)):
    
    # read the file
    image_bytes = await file.read()

    # open image using PIL
    image = Image.open(io.BytesIO(image_bytes))

    inputs = preprocessor(image, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_label = logits.argmax(-1).item()
    return JSONResponse(content={"predicted_bird": model.config.id2label[predicted_label]})