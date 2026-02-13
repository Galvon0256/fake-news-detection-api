import joblib
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

vectorizer = joblib.load(BASE_DIR / "vectorizer.jb")
model = joblib.load(BASE_DIR / "lr_model.jb")

classes = [
    "Fake",
    "Real"
]


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text


def predict_pipeline(text):

    transform_input = vectorizer.transform([text])
    prediction = model.predict(transform_input)

    result = classes[prediction[0]]

    return result
