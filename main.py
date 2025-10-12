import os
import json
import base64
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, Form, File
from fastapi.responses import JSONResponse
import torch
import torch.nn as nn
from datetime import datetime
from typing import List, Optional

# Placeholder Autoencoder (simplified for demo)
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 8, 3, stride=2, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Initialize model, load if exists
model = Autoencoder()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model_path = "model.pth"
if os.path.exists(model_path):
    try:
        model.load_state_dict(torch.load(model_path))
        model.eval()
        print(f"Loaded model from {model_path}")
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        print("Proceeding with a fresh model")
else:
    print(f"No model found at {model_path}. Proceeding with a fresh model")

# Load symptoms.json
with open("symptoms.json", "r") as f:
    symptoms_data = json.load(f)

# App setup
app = FastAPI()
DATA_DIR = "dataset"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "healthy"), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "diseased"), exist_ok=True)

# Load or initialize metadata
METADATA_FILE = "metadata.json"
if os.path.exists(METADATA_FILE):
    with open(METADATA_FILE, "r") as f:
        metadata = json.load(f)
else:
    metadata = []

# Preprocess image
def preprocess_image(image_bytes):
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (128, 128))
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return img, img_hsv

# Detect anomalies
def detect_anomaly(model, image_tensor):
    with torch.no_grad():
        reconstructed = model(image_tensor)
        mse = nn.functional.mse_loss(reconstructed, image_tensor).item()
    return mse, reconstructed

# Save image and metadata
def save_image_and_metadata(filename, label, error, image_bytes, user_input=None, symptom_match=None):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    if error > 0.05:  # Threshold for diseased
        save_path = os.path.join(DATA_DIR, "diseased", filename)
    else:
        save_path = os.path.join(DATA_DIR, "healthy", filename)
    
    with open(save_path, "wb") as f:
        f.write(image_bytes)
    
    metadata.append({
        "filename": filename,
        "label": label,
        "error": error,
        "timestamp": timestamp,
        "user_input": user_input,
        "symptom_match": symptom_match
    })
    
    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=2)

# API Endpoints
@app.post("/detect")
async def detect_disease(
    file: UploadFile = File(...),
    symptoms: str = Form(None),
    plant_age: str = Form(None),
    additional_notes: str = Form(None)
):
    # Read image bytes once
    image_bytes = await file.read()
    img, img_hsv = preprocess_image(image_bytes)

    # Convert to tensor
    img_tensor = torch.from_numpy(img.transpose((2, 0, 1))).float().unsqueeze(0).to(device) / 255.0
    error, reconstructed = detect_anomaly(model, img_tensor)

    # CV-based symptom matching
    symptom_match = None
    min_error = float('inf')
    for symptom, details in symptoms_data.items():
        lower, upper = np.array(details["hsv_range"][:3]), np.array(details["hsv_range"][3:])
        mask = cv2.inRange(img_hsv, lower, upper)
        if cv2.countNonZero(mask) > details["threshold"]:
            if error > details["threshold"] / 1000:  # Adjust threshold scale
                symptom_match = symptom
                break

    # Determine label
    label = "Healthy" if error < 0.05 else f"Diseased (Possible {symptom_match})" if symptom_match else "Diseased (Unknown)"
    user_symptoms = json.loads(symptoms) if symptoms else []

    # Generate filename with timestamp
    timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{file.filename.split('.')[0]}_{timestamp_str}.jpg"
    
    # Save image and metadata
    save_image_and_metadata(filename, label, error, image_bytes, user_symptoms, symptom_match)

    # Generate anomaly map (simplified)
    reconstructed_img = (reconstructed.cpu().squeeze(0).numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
    anomaly_map = cv2.absdiff(img, reconstructed_img)
    _, anomaly_map = cv2.threshold(anomaly_map, 30, 255, cv2.THRESH_BINARY)
    _, buffer = cv2.imencode('.jpg', anomaly_map)
    anomaly_map_base64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode()

    return JSONResponse({
        "label": label,
        "error": error,
        "anomaly_map": anomaly_map_base64,
        "detected_symptoms": [symptom_match] if symptom_match else []
    })

@app.post("/retrain")
async def retrain_model():
    # Load all images for retraining (simplified)
    images = []
    for folder in ["healthy", "diseased"]:
        folder_path = os.path.join(DATA_DIR, folder)
        if not os.path.exists(folder_path):
            continue
        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.resize(img, (128, 128))
            img_tensor = torch.from_numpy(img.transpose((2, 0, 1))).float().unsqueeze(0).to(device) / 255.0
            images.append(img_tensor)

    if images:
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        for epoch in range(10):  # Simple training loop
            for img in images:
                optimizer.zero_grad()
                output = model(img)
                loss = criterion(output, img)
                loss.backward()
                optimizer.step()
        torch.save(model.state_dict(), "model.pth")
        model.eval()
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        return JSONResponse({"status": f"Retraining completed at {timestamp}"})
    else:
        return JSONResponse({"status": "No images found for retraining"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)