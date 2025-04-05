from fastapi import FastAPI, UploadFile, File, HTTPException
import io
from PIL import Image
import numpy as np
import os

import torch
from ultralytics import YOLO

from tools.classifier import VitClassifier
import uvicorn

app = FastAPI()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MODELS_DIR = os.path.join(os.getcwd(), "models")  # Ensure your models are here
DETECTOR_PATH = os.path.join(MODELS_DIR, "detector.pt")

try:
    detector_model = YOLO(DETECTOR_PATH, task="detect")
    detector_model.to(device)
    detector_model.eval()
except Exception as e:
    raise RuntimeError(f"Failed to load detector model: {e}")

try:
    classifier_model = VitClassifier(MODELS_DIR, device, 0.2)
except Exception as e:
    raise RuntimeError(f"Failed to load classifier model: {e}")

@app.post("/inference")
async def inference(image: UploadFile = File(...)):
    try:
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    np_image = np.array(pil_image)

    try:
        results = detector_model([np_image], conf=0.25, imgsz=(768, 768))
        detection = results[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection failed: {e}")

    try:
        boxes = detection.boxes.xyxy.cpu().numpy().tolist()
    except Exception:
        boxes = []

    predictions = []
    transform = classifier_model.get_transformations()

    crops = []
    for bbox in boxes:
        x1, y1, x2, y2 = map(int, bbox)
        width, height = pil_image.size
        x1 = max(0, min(x1, width))
        x2 = max(0, min(x2, width))
        y1 = max(0, min(y1, height))
        y2 = max(0, min(y2, height))
        if x2 - x1 <= 0 or y2 - y1 <= 0:
            continue
        
        crop = np_image.copy()[y1:y2, x1:x2]
        crops.append(crop)
        
    transformed_crops = [Image.fromarray(transform(crop).numpy().transpose(1, 2, 0).astype(np.uint8)) for crop in crops]
    label_result = classifier_model([transformed_crops])[0]
    labels = list(map(lambda x: x[0], label_result))
    confidences = list(map(lambda x: x[1], label_result))
    
    return {"boxes": boxes,
        "labels": labels,
        "detector_confidences": [float(detection.boxes.conf[i]) for i in range(len(detection.boxes.conf))],
        "classifier_confidences": confidences}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)