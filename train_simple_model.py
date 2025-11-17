import cv2
import numpy as np
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

DATASET_PATH = 'kaggle-dataset/plantvillage dataset/color'
MODEL_PATH = 'simple_model.pkl'
LABEL_ENCODER_PATH = 'label_encoder.pkl'
IMAGE_SIZE = 128

def extract_features(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    
    hist_r = cv2.calcHist([img], [0], None, [32], [0, 256]).flatten()
    hist_g = cv2.calcHist([img], [1], None, [32], [0, 256]).flatten()
    hist_b = cv2.calcHist([img], [2], None, [32], [0, 256]).flatten()
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180]).flatten()
    hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256]).flatten()
    hist_v = cv2.calcHist([hsv], [2], None, [32], [0, 256]).flatten()
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean = np.mean(gray)
    std = np.std(gray)
    
    features = np.concatenate([
        hist_r, hist_g, hist_b,
        hist_h, hist_s, hist_v,
        [mean, std]
    ])
    
    return features

def load_dataset():
    print("Loading dataset...")
    
    features = []
    labels = []
    
    class_folders = [f for f in os.listdir(DATASET_PATH) 
                    if os.path.isdir(os.path.join(DATASET_PATH, f))]
    
    print(f"Found {len(class_folders)} classes")
    
    for class_name in tqdm(class_folders, desc="Processing classes"):
        class_path = os.path.join(DATASET_PATH, class_name)
        
        images = [f for f in os.listdir(class_path) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:100]
        
        for img_name in images:
            img_path = os.path.join(class_path, img_name)
            
            feat = extract_features(img_path)
            if feat is not None:
                features.append(feat)
                labels.append(class_name)
    
    return np.array(features), np.array(labels)

def train_model():
    print("\nTraining ML Model\n")
    
    X, y = load_dataset()
    print(f"\nLoaded {len(X)} samples from {len(np.unique(y))} classes")
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    print("\nTraining Random Forest...")
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    clf.fit(X_train, y_train)
    
    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)
    
    print(f"\nTraining Accuracy: {train_acc:.2%}")
    print(f"Test Accuracy: {test_acc:.2%}")
    
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(clf, f)
    
    with open(LABEL_ENCODER_PATH, 'wb') as f:
        pickle.dump(le, f)
    
    print(f"\nModel saved: {MODEL_PATH}")
    print(f"Label encoder saved: {LABEL_ENCODER_PATH}")
    print(f"\nRun: python main_simple.py")
    
    return clf, le

if __name__ == '__main__':
    if not os.path.exists(DATASET_PATH):
        print(f"ERROR: Dataset not found at {DATASET_PATH}")
        exit(1)
    
    train_model()
