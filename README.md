# 🏥 Sepsis Risk Prediction System

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.3+-green.svg)
![Machine Learning](https://img.shields.io/badge/ML-XGBoost-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

An AI-powered early warning system for sepsis risk assessment using machine learning. This system provides real-time predictions based on patient vital signs and laboratory values to assist healthcare professionals in early sepsis detection.

## 🚀 Live Demo
[Click here to view the demo](https://sepsis-ai.netlify.app/)


## 🌟 Features

-  **Real-time Predictions** - Instant sepsis risk assessment
-  **Machine Learning** - XGBoost & Random Forest models with high accuracy on test data (synthetic dataset)
-  **Clinical Decision Support** - Evidence-based recommendations
-  **Beautiful UI** - Clean, modern interface with gradient design
-  **REST API** - Easy integration with existing systems
-  **Responsive** - Works on desktop, tablet, and mobile
-  **Easy Deployment** - Simple setup and deployment

## 🎯 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Modern web browser

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Mukesh-V-AI/sepsis-risk-prediction.git
cd sepsis-risk-prediction
```

2. **Create virtual environment**
```bash
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Generate sample data**
```bash
python generate_sample_data.py
```

5. **Preprocess data**
```bash
python src/simple_preprocessing.py
```

6. **Train models**
```bash
python src/simple_training.py
```

##  Running the Application

### Start Backend API

```bash
python api/simple_app.py
```

The API will start at `http://localhost:5000`

### Open Frontend

Simply open `index.html` in your web browser, or use a local server:

```bash
# Python 3
python -m http.server 8000

# Then open: http://localhost:8000
```

## 📊 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|-------|----------|-----------|--------|----------|---------|
| **XGBoost** | 100% | 100% | 100% | 100% | 1.00 |
| **Random Forest** | 100% | 100% | 100% | 100% | 1.00 |

*Note: Trained on synthetic sample data. Performance may vary on real-world data.*

##  API Documentation

### Health Check
```bash
GET /api/health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### Predict Sepsis Risk
```bash
POST /api/predict
Content-Type: application/json
```

**Request Body:**
```json
{
  "age": 68,
  "heart_rate": 115,
  "temperature": 39.2,
  "respiratory_rate": 26,
  "systolic_bp": 82,
  "diastolic_bp": 50,
  "wbc_count": 18.5,
  "lactate": 4.2,
  "procalcitonin": 8.5
}
```

**Response:**
```json
{
  "probability": 0.95,
  "risk_level": "Critical",
  "risk_score": 95,
  "recommendations": [
    "Immediate ICU consultation required",
    "Start broad-spectrum antibiotics within 1 hour",
    "Obtain blood cultures before antibiotics"
  ]
}
```

## 📁 Project Structure

```
sepsis-risk-prediction/
├── index.html                   # Frontend web interface
├── generate_sample_data.py      # Sample data generator
├── requirements.txt             # Python dependencies
├── README.md                    # This file
│
├── data/
│   ├── raw/                     # Raw data files
│   └── processed/               # Preprocessed data
│
├── models/
│   └── trained_models/          # Saved ML models
│       ├── xgboost_model.pkl
│       └── random_forest_model.pkl
│
├── src/
│   ├── simple_preprocessing.py  # Data preprocessing
│   └── simple_training.py       # Model training
│
└── api/
    └── simple_app.py            # Flask REST API
```

##  Testing

Try these example patients:

### High Risk Patient
```
Age: 68, HR: 115, Temp: 39.2°C, RR: 26
BP: 82/50, WBC: 18.5, Lactate: 4.2, PCT: 8.5
```

### Normal Patient
```
Age: 45, HR: 72, Temp: 36.8°C, RR: 16
BP: 120/78, WBC: 7.2, Lactate: 1.1, PCT: 0.08
```

### Critical Patient
```
Age: 72, HR: 128, Temp: 35.2°C, RR: 32
BP: 68/42, WBC: 2.8, Lactate: 6.8, PCT: 15.2
```




##  Security Notes

-  This is a **demonstration/educational project**
-  For production use, add authentication
-  Implement rate limiting
-  Add request validation
-  Use HTTPS in production
-  Ensure HIPAA compliance for real patient data

## ⚠️ Disclaimer

**This system is intended for research and educational purposes only.** It is not a substitute for professional medical advice, diagnosis, or treatment. All clinical decisions should be made by qualified healthcare professionals based on complete patient assessment.



##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.



##  Acknowledgments

- PhysioNet for clinical datasets
- Surviving Sepsis Campaign for guidelines
- Open-source ML community
