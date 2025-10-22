import os
import json
import base64
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, Form, File
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
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

# Initialize device first
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize model
model = Autoencoder()
model = model.to(device)
model_path = "model.pth"
if os.path.exists(model_path):
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print(f"Loaded model from {model_path}")
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        print("Proceeding with a fresh model")
else:
    print(f"No model found at {model_path}. Proceeding with a fresh model")

# Load symptoms.json with error handling
symptoms_data = {}
if os.path.exists("symptoms.json"):
    try:
        with open("symptoms.json", "r") as f:
            symptoms_data = json.load(f)
        print("Loaded symptoms.json")
    except Exception as e:
        print(f"Error loading symptoms.json: {e}")
        print("Proceeding without symptom matching")
else:
    print("symptoms.json not found. Creating default version...")
    default_symptoms = {
        "leaf_spot": {
            "hsv_range": [0, 50, 50, 30, 255, 255],
            "threshold": 1000
        },
        "yellowing": {
            "hsv_range": [20, 100, 100, 40, 255, 255],
            "threshold": 1500
        },
        "blight": {
            "hsv_range": [0, 0, 0, 180, 50, 100],
            "threshold": 2000
        }
    }
    with open("symptoms.json", "w") as f:
        json.dump(default_symptoms, f, indent=2)
    symptoms_data = default_symptoms
    print("Created default symptoms.json")

# App setup
app = FastAPI()

# NO CORS NEEDED! Serve static files from same origin
# Create static directory if it doesn't exist
os.makedirs("static", exist_ok=True)

DATA_DIR = "dataset"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "healthy"), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "diseased"), exist_ok=True)

# Load or initialize metadata
METADATA_FILE = "metadata.json"
if os.path.exists(METADATA_FILE):
    try:
        with open(METADATA_FILE, "r") as f:
            metadata = json.load(f)
    except Exception as e:
        print(f"Error loading metadata: {e}")
        metadata = []
else:
    metadata = []

# Preprocess image
def preprocess_image(image_bytes):
    try:
        img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image")
        img = cv2.resize(img, (128, 128))
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        return img, img_hsv
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        raise

# Detect anomalies
def detect_anomaly(model, image_tensor):
    with torch.no_grad():
        reconstructed = model(image_tensor)
        mse = nn.functional.mse_loss(reconstructed, image_tensor).item()
    return mse, reconstructed

# Save image and metadata
def save_image_and_metadata(filename, label, error, image_bytes, user_input=None, symptom_match=None):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if error > 0.05:
        save_path = os.path.join(DATA_DIR, "diseased", filename)
    else:
        save_path = os.path.join(DATA_DIR, "healthy", filename)
    
    with open(save_path, "wb") as f:
        f.write(image_bytes)
    
    metadata.append({
        "filename": filename,
        "label": label,
        "error": float(error),
        "timestamp": timestamp,
        "user_input": user_input,
        "symptom_match": symptom_match
    })
    
    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=2)

# Root endpoint - serve the HTML frontend
@app.get("/", response_class=HTMLResponse)
async def root():
    if os.path.exists("static/index.html"):
        with open("static/index.html", "r") as f:
            return f.read()
    return """
    <html>
        <head><title>Plant Disease Detection</title></head>
        <body>
            <h1>Plant Disease Detection API</h1>
            <p>API is running! Place your index.html in the /static folder</p>
            <p><a href="/docs">View API Documentation</a></p>
        </body>
    </html>
    """

# API Endpoints (prefixed with /api to separate from static content)
@app.post("/api/detect")
async def detect_disease(
    file: UploadFile = File(...),
    symptoms: str = Form(None),
    plant_age: str = Form(None),
    additional_notes: str = Form(None)
):
    try:
        image_bytes = await file.read()
        img, img_hsv = preprocess_image(image_bytes)

        img_tensor = torch.from_numpy(img.transpose((2, 0, 1))).float().unsqueeze(0).to(device) / 255.0
        error, reconstructed = detect_anomaly(model, img_tensor)

        symptom_match = None
        if symptoms_data:
            for symptom, details in symptoms_data.items():
                lower = np.array(details["hsv_range"][:3], dtype=np.uint8)
                upper = np.array(details["hsv_range"][3:], dtype=np.uint8)
                mask = cv2.inRange(img_hsv, lower, upper)
                pixel_count = cv2.countNonZero(mask)
                
                if pixel_count > details["threshold"] and error > 0.03:
                    symptom_match = symptom
                    break

        label = "Healthy" if error < 0.05 else f"Diseased (Possible {symptom_match})" if symptom_match else "Diseased (Unknown)"
        
        user_symptoms = []
        if symptoms:
            try:
                user_symptoms = json.loads(symptoms)
            except:
                user_symptoms = [symptoms]

        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        original_name = file.filename.split('.')[0] if '.' in file.filename else file.filename
        filename = f"{original_name}_{timestamp_str}.jpg"
        
        save_image_and_metadata(filename, label, error, image_bytes, user_symptoms, symptom_match)

        reconstructed_img = (reconstructed.cpu().squeeze(0).numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
        anomaly_map = cv2.absdiff(img, reconstructed_img)
        _, anomaly_map = cv2.threshold(anomaly_map, 30, 255, cv2.THRESH_BINARY)
        _, buffer = cv2.imencode('.jpg', anomaly_map)
        anomaly_map_base64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode()

        return JSONResponse({
            "label": label,
            "error": float(error),
            "anomaly_map": anomaly_map_base64,
            "detected_symptoms": [symptom_match] if symptom_match else []
        })
    except Exception as e:
        print(f"Error in detect_disease: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/api/retrain")
async def retrain_model():
    try:
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
            
            print(f"Starting retraining with {len(images)} images...")
            for epoch in range(10):
                total_loss = 0
                for img in images:
                    optimizer.zero_grad()
                    output = model(img)
                    loss = criterion(output, img)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                print(f"Epoch {epoch+1}/10, Loss: {total_loss/len(images):.4f}")
            
            torch.save(model.state_dict(), "model.pth")
            model.eval()
            
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            return JSONResponse({
                "status": f"Retraining completed at {timestamp}",
                "images_used": len(images)
            })
        else:
            return JSONResponse({"status": "No images found for retraining"}, status_code=400)
    except Exception as e:
        print(f"Error in retrain_model: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/api/metadata")
async def get_metadata():
    """Get all stored metadata"""
    return JSONResponse(metadata)

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "device": str(device)}

# Mount static files AFTER defining all routes
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)