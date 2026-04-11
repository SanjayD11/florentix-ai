# 🌱 Plant Disease Prediction AI

An AI-powered web application that detects tomato plant diseases from leaf images using Deep Learning.  
Users can upload a plant leaf image and instantly get disease prediction, confidence score, and treatment recommendations.

---

## 🚀 Features

✅ Upload plant leaf image  
✅ AI disease detection using Deep Learning  
✅ Confidence score prediction  
✅ Treatment & prevention suggestions  
✅ Fast inference using TensorFlow Lite  
✅ REST API with FastAPI backend  
✅ User-friendly web interface  

---

## 🧠 Model Details

### Architecture
- Transfer Learning
- MobileNetV2 (ImageNet Pretrained)
- Custom classification head

### Training Strategy
- **Phase 1 — Feature Extraction**
  - Base MobileNetV2 frozen
  - Only custom layers trained

- **Phase 2 — Fine Tuning**
  - Top layers of MobileNetV2 unfrozen
  - Lower learning rate used
  - Improved accuracy

### Dataset
Plant leaf disease dataset containing 5 tomato disease classes:

- Early Blight
- Healthy
- Late Blight
- Leaf Mold
- Septoria Leaf Spot

### Performance
- **Test Accuracy:** ~88%
- Image size: 224×224
- Model optimized using TensorFlow Lite for faster inference

---

## 🏗️ Project Structure
```
project/
│
├── backend/ # FastAPI server
│ ├── app.py
│ └── utils/
│ ├── image_utils.py
│ └── remedies.py
│
├── frontend/ # Web UI (HTML, CSS, JS)
│
├── dataset/ # Training images (not included)
│
├── model/ # Trained model files (.h5, .tflite)
│
├── train_model.ipynb # Model training notebook
│
└── convert_model.py # TensorFlow Lite conversion script


```  
---

## ⚙️ Tech Stack

### AI / ML
- TensorFlow / Keras
- MobileNetV2
- Transfer Learning
- TensorFlow Lite
- NumPy
- OpenCV

### Backend
- FastAPI
- Python

### Frontend
- HTML
- CSS
- JavaScript

### Tools
- VS Code
- Git & GitHub
- Jupyter Notebook

---

## 📊 Model Training Pipeline

### 1️⃣ Data Preparation
- Dataset split → Train / Validation / Test
- Data augmentation:
  - Rotation
  - Zoom
  - Horizontal flip
  - Width/Height shift

### 2️⃣ Transfer Learning
- MobileNetV2 pretrained weights
- Feature extraction training

### 3️⃣ Fine Tuning
- Unfreezing top layers
- Lower learning rate
- Improved accuracy

### 4️⃣ Model Optimization
- TensorFlow Lite conversion
- Reduced model size
- Faster prediction speed

---

## 🧪 API Endpoints

### Health Check
GET /


Response:
{
"status": "running",
"message": "Plant Disease API is live"
}


---

### Predict Disease
POST /predict


Upload image → Returns:

{
"prediction": "Early_Blight",
"confidence": 88.4
}


---

## 💻 Local Setup

### 1. Clone Repository
git clone https://github.com/siddhisarode/plant-disease-prediction-ai.git
cd plant-disease-prediction-ai


### 2. Create Virtual Environment
python -m venv plantenv
plantenv\Scripts\activate


### 3. Install Dependencies
pip install -r requirements.txt


### 4. Run Backend Server
uvicorn backend.app:app --reload


Backend runs at:
http://127.0.0.1:8000


### 5. Run Frontend
Open the frontend folder using Live Server or any local web server.

---

## 🎯 Use Cases

- Smart agriculture systems
- Plant disease detection automation
- Crop monitoring tools
- AI-powered gardening assistant
- Household plant care guidance

---

## 📈 Future Improvements

- Support for more plant species
- Higher accuracy with larger dataset
- Cloud deployment
- Mobile app integration
- Real-time camera detection
- Weather-based plant care suggestions

---

## 👨‍💻 Author

**Siddhi Sarode**  
AI / ML Developer — Python, Computer Vision & Deep Learning

---

## ⭐ If you like this project, give it a star!

