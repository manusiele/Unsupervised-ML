Crop Disease Detection API
A Python-based backend API for unsupervised crop disease detection using computer vision (OpenCV) and machine learning (PyTorch). The app processes crop images, detects diseases as anomalies (e.g., powdery mildew), and saves images for iterative model improvement. Built with FastAPI for global access, it uses a file system and JSON metadata for storage.
Features

Unsupervised Detection: Uses an autoencoder to detect diseases as anomalies (high reconstruction error) without labeled data.
Computer Vision: Preprocesses images with OpenCV (resize, segmentation, color enhancement based on JSON symptoms).
API Endpoints: /detect for image analysis (returns label, error, anomaly heatmap); /retrain for model updates.
Iterative Improvement: Saves user-uploaded images to dataset/ and metadata to metadata.json for retraining.
Lightweight Storage: File system (dataset/healthy/, dataset/diseased/) with JSON metadata, no database required.

Requirements

Python 3.9+
Libraries: fastapi, uvicorn, opencv-python, torch, torchvision, numpy, pillow, python-multipart
Initial dataset: 50-100 healthy crop images (e.g., from PlantVillage)
symptoms.json file for symptom guidance (e.g., {"healthy": {"color": "green"}, "powdery_mildew": {"color": "white"}})

Installation

Clone the repository:git clone https://github.com/manusiele/crop-disease-app.git
cd crop-disease-app


Create and activate a virtual environment:python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate


Install dependencies:pip install fastapi uvicorn opencv-python torch torchvision numpy pillow python-multipart


Download PlantVillage dataset (healthy images) from Kaggle and place in dataset/healthy/.
Ensure symptoms.json is in the root directory.
(Optional) Place a pre-trained model (model.pth) in the root, or train one (see Usage).

Usage

Run the API Locally:uvicorn main:app --reload

API runs at http://localhost:8000.
Test the API:
Use Postman or curl to send an image to http://localhost:8000/detect.
Example:curl -X POST -F "file=@test_image.jpg" http://localhost:8000/detect

Response: JSON with label (e.g., "Diseased (Possible Powdery Mildew)"), error, and base64-encoded anomaly_map.


Save Images: Confirm results to save images to dataset/healthy/ or dataset/diseased/ with metadata in metadata.json.
Retrain Model: Call /retrain endpoint to update the model with saved images.

API Endpoints

POST /detect: Upload an image, get detection results.
Input: Image file (jpg/png)
Output: JSON { "label": str, "error": float, "anomaly_map": base64 }


POST /retrain: Retrain the model on saved images.
Input: None
Output: JSON { "status": "Retraining completed" }



Dataset

Initial Training: Uses 50-100 healthy images from PlantVillage (Kaggle). Place in dataset/healthy/.
Iterative Data: User-uploaded images saved to dataset/ for retraining.
Metadata: Stored in metadata.json (e.g., filename, label, error, timestamp).

Project Structure
crop-disease-app/
├── main.py            # FastAPI app with CV/ML logic
├── symptoms.json      # Symptom descriptions for CV
├── model.pth         # Pre-trained autoencoder model
├── dataset/
│   ├── healthy/      # Healthy crop images
│   ├── diseased/     # Diseased crop images
│   └── metadata.json # Metadata for saved images
└── README.md         # Project documentation

Future Improvements

Add authentication for API endpoints (e.g., API keys).
Integrate cloud storage (e.g., AWS S3) for scalability.
Support real-time camera input via OpenCV.
Use clustering (e.g., K-means) for refined disease grouping.

Acknowledgments

Inspired by hackathon projects like AgriDoc AI and Maizpert.
Datasets: PlantVillage, PlantDoc.
Tools: FastAPI, OpenCV, PyTorch.
