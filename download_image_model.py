from transformers import AutoImageProcessor, SiglipForImageClassification

MODEL_NAME = "prithivMLmods/deepfake-detector-model-v1"
SAVE_PATH = "./local_deepfake_model"

model = SiglipForImageClassification.from_pretrained(MODEL_NAME)
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

model.save_pretrained(SAVE_PATH)
processor.save_pretrained(SAVE_PATH)

print("Model saved locally!")
