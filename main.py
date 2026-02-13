from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel

# Existing text model pipeline
from models.models import predict_pipeline

# Image model imports
import torch
from transformers import AutoImageProcessor, SiglipForImageClassification
from PIL import Image
import io


# ------------------- FASTAPI APP -------------------
app = FastAPI(title="Multi-Model Prediction API")


# ------------------- TEXT SCHEMA -------------------
class TextIn(BaseModel):
    text: str


class TextPredictionOut(BaseModel):
    output: str


# ------------------- IMAGE SCHEMA -------------------
class ImagePredictionOut(BaseModel):
    fake: float
    real: float


# ------------------- LOAD IMAGE MODEL (ONCE) -------------------

MODEL_PATH = "./local_deepfake_model"

ID2LABEL = {
    "0": "fake",
    "1": "real"
}

device = "cuda" if torch.cuda.is_available() else "cpu"

# print(f"Loading image model on device: {device}")

# image_model = SiglipForImageClassification.from_pretrained(MODEL_PATH)
# image_processor = AutoImageProcessor.from_pretrained(MODEL_PATH)

# image_model.to(device)
# image_model.eval()


image_model = None
image_processor = None

@app.on_event("startup")
def load_models():
    global image_model, image_processor

    print(f"Loading image model on device: {device}")

    image_model = SiglipForImageClassification.from_pretrained(MODEL_PATH)
    image_processor = AutoImageProcessor.from_pretrained(MODEL_PATH)

    image_model.to(device)
    image_model.eval()


# ------------------- IMAGE PREDICTION FUNCTION -------------------

def classify_image_bytes(image_bytes: bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    inputs = image_processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = image_model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()

    prediction = {
        ID2LABEL[str(i)]: round(float(probs[i]), 3)
        for i in range(len(probs))
    }

    return prediction


# ------------------- ROUTES -------------------

@app.get("/")
def home():
    return {"health_check": "OK"}


# ---------- TEXT PREDICTION ENDPOINT ----------
@app.post("/predict-text", response_model=TextPredictionOut)
def predict_text(payload: TextIn):
    output = predict_pipeline(payload.text)
    return {"output": output}


# ---------- IMAGE PREDICTION ENDPOINT ----------
@app.post("/predict-image", response_model=ImagePredictionOut)
async def predict_image(file: UploadFile = File(...)):

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    image_bytes = await file.read()

    prediction = classify_image_bytes(image_bytes)

    return prediction


# ------------------- RUN SERVER -------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app="main:app", host="0.0.0.0", port=8000, reload=True)
