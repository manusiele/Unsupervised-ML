import os
import json
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
if os.path.exists("model.pth"):
    model.load_state_dict(torch.load("model.pth"))
    model.eval()

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
def preprocess_image(image):
    img = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
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
def save_image_and_metadata(filename, label, error, user_input=None, symptom_match=None):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    if error > 0.05:  # Threshold for diseased
        save_path = os.path.join(DATA_DIR, "diseased", filename)
    else:
        save_path = os.path.join(DATA_DIR, "healthy", filename)
    with open(save_path, "wb") as f:
        f.write(image_file.file.read())  # Reopen file to read
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
    global image_file
    image_file = file
    img, img_hsv = preprocess_image(await file.read())

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

    # Save image and metadata
    filename = f"{file.filename.split('.')[0]}_{timestamp.replace(':', '-')}.jpg"
    save_image_and_metadata(filename, label, error, user_symptoms, symptom_match)

    # Generate anomaly map (simplified)
    anomaly_map = cv2.absdiff(img, (reconstructed.cpu().squeeze(0).numpy().transpose((1, 2, 0)) * 255).astype(np.uint8))
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
        for filename in os.listdir(os.path.join(DATA_DIR, folder)):
            img_path = os.path.join(DATA_DIR, folder, filename)
            img = cv2.imread(img_path)
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

    return JSONResponse({"status": "Retraining completed at 2025-10-01 14:31"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)