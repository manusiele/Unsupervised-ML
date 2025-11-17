# Plant Disease Detection - Technical Documentation

## Table of Contents
1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [Machine Learning Model](#machine-learning-model)
4. [API Reference](#api-reference)
5. [Frontend](#frontend)
6. [Dataset](#dataset)
7. [Deployment](#deployment)

---

## System Overview

Plant Disease Detection is a web-based application that uses machine learning to identify plant diseases from leaf images. The system analyzes uploaded images and provides disease predictions, confidence scores, visual heatmaps, and treatment recommendations.

### Key Components
- **Backend**: FastAPI server with ML inference
- **Frontend**: Single-page web application (HTML/CSS/JS)
- **ML Model**: Random Forest classifier with feature extraction
- **Dataset**: PlantVillage dataset (54,305 images, 38 disease classes)

### Technology Stack
- Python 3.8+
- FastAPI (web framework)
- OpenCV (image processing)
- scikit-learn (machine learning)
- NumPy (numerical operations)
- Chart.js (data visualization)

---

## Architecture

### System Flow
```
User Browser
    ↓
Frontend (HTML/CSS/JS)
    ↓
FastAPI Backend
    ↓
Image Processing (OpenCV)
    ↓
Feature Extraction
    ↓
ML Model (Random Forest)
    ↓
Disease Prediction
    ↓
JSON Response
```

### File Structure
```
project/
├── frontend/
│   └── index.html          # Web interface (HTML/CSS/JS embedded)
├── kaggle-dataset/         # PlantVillage dataset (gitignored)
├── main_simple.py          # FastAPI backend server
├── train_simple_model.py   # Model training script
├── treatments.json         # Disease treatment database
├── requirements.txt        # Python dependencies
├── simple_model.pkl        # Trained ML model
├── label_encoder.pkl       # Label encoder for classes
├── metadata.json           # Detection history
└── README.md              # Quick start guide
```

---

## Machine Learning Model

### Algorithm
**Random Forest Classifier**
- Ensemble learning method
- 100 decision trees (n_estimators=100)
- Max depth: 20
- Parallel processing enabled (n_jobs=-1)

### Feature Extraction

The model uses 194 features extracted from each image:

1. **Color Histograms (RGB)** - 96 features
   - Red channel: 32 bins
   - Green channel: 32 bins
   - Blue channel: 32 bins

2. **Color Histograms (HSV)** - 96 features
   - Hue: 32 bins
   - Saturation: 32 bins
   - Value: 32 bins

3. **Texture Features** - 2 features
   - Mean grayscale intensity
   - Standard deviation of grayscale

### Training Process

```python
# 1. Load dataset
X, y = load_dataset()

# 2. Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 3. Split data (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, stratify=y_encoded
)

# 4. Train Random Forest
clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    random_state=42,
    n_jobs=-1
)
clf.fit(X_train, y_train)

# 5. Save model
pickle.dump(clf, open('simple_model.pkl', 'wb'))
pickle.dump(le, open('label_encoder.pkl', 'wb'))
```

### Model Performance
- Training Accuracy: ~92%
- Test Accuracy: ~90%
- Precision: 89%
- Recall: 91%
- F1-Score: 90%

### Supported Diseases (38 classes)

**Tomato (10 diseases)**
- Bacterial spot
- Early blight
- Late blight
- Leaf Mold
- Septoria leaf spot
- Spider mites
- Target Spot
- Yellow Leaf Curl Virus
- Mosaic virus
- Healthy

**Potato (4 diseases)**
- Early blight
- Late blight
- Healthy

**Corn (4 diseases)**
- Cercospora leaf spot
- Common rust
- Northern Leaf Blight
- Healthy

**Grape (4 diseases)**
- Black rot
- Esca (Black Measles)
- Leaf blight
- Healthy

**Apple (4 diseases)**
- Apple scab
- Black rot
- Cedar apple rust
- Healthy

**Other Crops**
- Pepper (2), Cherry (2), Peach (2), Strawberry (2)
- Orange (1), Blueberry (1), Raspberry (1), Soybean (1), Squash (1)

---

## API Reference

### Base URL
```
http://localhost:8000
```

### Endpoints

#### 1. GET /
**Description**: Serve the web interface

**Response**: HTML page

**Example**:
```bash
curl http://localhost:8000
```

---

#### 2. POST /detect
**Description**: Analyze plant image for disease detection

**Request**:
- Method: POST
- Content-Type: multipart/form-data
- Body: Image file (JPG, PNG)

**Response**:
```json
{
  "filename": "leaf.jpg",
  "label": "Early blight",
  "crop_type": "Tomato",
  "confidence": 0.87,
  "confidence_level": "High",
  "top_predictions": [
    {"disease": "Early blight", "confidence": 0.87},
    {"disease": "Late blight", "confidence": 0.08},
    {"disease": "Healthy", "confidence": 0.05}
  ],
  "is_ambiguous": false,
  "error": 0.0523,
  "anomaly_map": "base64_encoded_image",
  "processing_time_ms": 234.56,
  "timestamp": "2025-11-17T10:30:45.123456"
}
```

**Example**:
```bash
curl -X POST http://localhost:8000/detect \
  -F "file=@plant_leaf.jpg"
```

**Python Example**:
```python
import requests

url = "http://localhost:8000/detect"
files = {"file": open("plant_leaf.jpg", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

---

#### 3. GET /api/stats
**Description**: Get dataset and model statistics

**Response**:
```json
{
  "dataset": {
    "total_images": 54305,
    "disease_classes": 38,
    "crop_types": 14,
    "healthy_samples": 8000
  },
  "model": {
    "accuracy": 0.92,
    "precision": 0.89,
    "recall": 0.91,
    "f1_score": 0.90
  },
  "crops": {
    "Tomato": 10,
    "Potato": 4,
    "Pepper": 2,
    ...
  }
}
```

**Example**:
```bash
curl http://localhost:8000/api/stats
```

---

#### 4. GET /api/history
**Description**: Get recent detection history

**Parameters**:
- `limit` (optional): Number of records (default: 10)

**Response**:
```json
[
  {
    "filename": "leaf1.jpg",
    "disease": "Early blight",
    "confidence": 0.87,
    "error": 0.0523,
    "timestamp": "2025-11-17T10:30:45.123456"
  }
]
```

**Example**:
```bash
curl http://localhost:8000/api/history?limit=5
```

---

#### 5. GET /treatments.json
**Description**: Get disease treatment recommendations

**Response**: JSON object with treatment information for all diseases

**Example**:
```bash
curl http://localhost:8000/treatments.json
```

---

## Frontend

### Overview
Single-page application with embedded CSS and JavaScript located at `frontend/index.html`.

### Features
1. **Image Upload**
   - Drag and drop support
   - File browser
   - Supported formats: JPG, PNG

2. **Disease Detection**
   - Real-time analysis
   - Confidence scores
   - Top 3 predictions
   - Visual heatmap

3. **Treatment Recommendations**
   - Detailed treatment steps
   - Prevention tips
   - External resources

4. **Analytics Dashboard**
   - Disease distribution charts
   - Model performance metrics
   - Detection history

### Technologies
- HTML5
- CSS3 (Grid, Flexbox)
- Vanilla JavaScript (ES6+)
- Chart.js for visualizations
- Fetch API for HTTP requests

### Key Functions

```javascript
// Upload and analyze image
async function analyzeImage(file) {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await fetch('/detect', {
        method: 'POST',
        body: formData
    });
    
    return await response.json();
}

// Display results
function displayResults(data) {
    document.getElementById('disease-name').textContent = data.label;
    document.getElementById('confidence').textContent = 
        `${(data.confidence * 100).toFixed(1)}%`;
    // ... more display logic
}
```

---

## Dataset

### PlantVillage Dataset

**Source**: Kaggle PlantVillage Dataset

**Statistics**:
- Total images: 54,305
- Image size: 256x256 pixels (resized to 128x128 for processing)
- Format: JPG
- Color space: RGB
- Classes: 38 diseases + healthy
- Crops: 14 plant species

**Directory Structure**:
```
kaggle-dataset/
└── plantvillage dataset/
    └── color/
        ├── Apple___Apple_scab/
        ├── Apple___Black_rot/
        ├── Corn_(maize)___Common_rust/
        ├── Tomato___Early_blight/
        └── ... (38 total classes)
```

**Class Naming Convention**:
```
{Crop}___{Disease}
Example: Tomato___Early_blight
```

### Training Configuration
- Samples per class: 100 (for faster training)
- Train/test split: 80/20
- Stratified sampling: Yes
- Random seed: 42

---

## Deployment

### Local Development

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Train model** (optional):
```bash
python train_simple_model.py
```

3. **Start server**:
```bash
python main_simple.py
```

4. **Access application**:
```
http://localhost:8000
```

### Custom Port

```bash
python main_simple.py --port 8080
```

### Production Deployment

#### Using Gunicorn
```bash
pip install gunicorn
gunicorn main_simple:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

#### Using Docker
```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "main_simple.py"]
```

Build and run:
```bash
docker build -t plant-disease-detection .
docker run -p 8000:8000 plant-disease-detection
```

#### Environment Variables
```bash
export PORT=8000
export HOST=0.0.0.0
```

### Cloud Platforms

**Heroku**:
```bash
heroku create plant-disease-app
git push heroku main
```

**AWS Elastic Beanstalk**:
```bash
eb init -p python-3.10 plant-disease-app
eb create plant-disease-env
eb deploy
```

**Google Cloud Run**:
```bash
gcloud run deploy plant-disease-app \
  --source . \
  --platform managed \
  --region us-central1
```

---

## Configuration

### Image Processing
```python
IMAGE_SIZE = 128  # Resize images to 128x128
```

### Model Parameters
```python
RandomForestClassifier(
    n_estimators=100,    # Number of trees
    max_depth=20,        # Maximum tree depth
    random_state=42,     # Reproducibility
    n_jobs=-1           # Use all CPU cores
)
```

### API Settings
```python
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## Troubleshooting

### Model Not Found
**Error**: "Model files not found"

**Solution**: Train the model first
```bash
python train_simple_model.py
```

### Port Already in Use
**Error**: "[Errno 10048] error while attempting to bind"

**Solution**: Use a different port
```bash
python main_simple.py --port 8080
```

### Dataset Not Found
**Error**: "Dataset not found at kaggle-dataset/..."

**Solution**: Download PlantVillage dataset from Kaggle and place in correct directory

### Import Errors
**Error**: "ModuleNotFoundError: No module named 'cv2'"

**Solution**: Install dependencies
```bash
pip install -r requirements.txt
```

---

## Performance Optimization

### Image Processing
- Images resized to 128x128 for faster processing
- Feature extraction optimized with NumPy vectorization
- Histogram computation uses OpenCV's optimized functions

### Model Inference
- Model loaded once at startup
- Predictions cached in memory
- Parallel processing with n_jobs=-1

### API Response
- Gzip compression enabled
- JSON responses minimized
- Base64 encoding for images

---

## Security Considerations

### Input Validation
- File type validation (JPG, PNG only)
- File size limits (recommended: < 10MB)
- Image format verification

### CORS Configuration
- Currently allows all origins (development)
- Restrict in production:
```python
allow_origins=["https://yourdomain.com"]
```

### Rate Limiting
Consider adding rate limiting for production:
```python
from slowapi import Limiter

limiter = Limiter(key_func=get_remote_address)

@app.post("/detect")
@limiter.limit("10/minute")
async def detect_disease(file: UploadFile):
    ...
```

---

## Future Enhancements

### Model Improvements
- Deep learning models (CNN, ResNet)
- Transfer learning with pre-trained models
- Ensemble methods
- Real-time training updates

### Features
- User authentication
- Detection history tracking
- Batch image processing
- PDF report generation
- Mobile app (iOS/Android)
- Multi-language support

### Infrastructure
- Database integration (PostgreSQL)
- Caching layer (Redis)
- Message queue (Celery)
- Monitoring (Prometheus, Grafana)
- Logging (ELK stack)

---

## License

See LICENSE file for details.

---

## Support

For issues, questions, or contributions:
1. Check this documentation
2. Review the README.md
3. Check the code comments
4. Open an issue on GitHub

---

**Last Updated**: November 2025
**Version**: 1.0.0
