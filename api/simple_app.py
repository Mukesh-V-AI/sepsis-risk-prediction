"""
Simple Flask API for Sepsis Prediction
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)
CORS(app)

# Get the correct path
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(base_dir, 'models', 'trained_models', 'xgboost_model.pkl')

print(f"Loading model from: {model_path}")
model = joblib.load(model_path)
print("✅ Model loaded successfully!")

def calculate_risk_level(probability):
    """Determine risk level from probability"""
    if probability >= 0.7:
        return 'Critical', 'red'
    elif probability >= 0.5:
        return 'High', 'orange'
    elif probability >= 0.3:
        return 'Moderate', 'yellow'
    else:
        return 'Low', 'green'

def generate_recommendations(risk_level, patient_data):
    """Generate clinical recommendations"""
    recommendations = []
    
    if risk_level == 'Critical':
        recommendations = [
            'Immediate ICU consultation required',
            'Start broad-spectrum antibiotics within 1 hour',
            'Obtain blood cultures before antibiotics',
            'Continuous monitoring of vital signs',
            'Consider vasopressor support if needed'
        ]
    elif risk_level == 'High':
        recommendations = [
            'Close monitoring required',
            'Consider early antibiotic therapy',
            'Repeat assessment in 2-4 hours',
            'Watch for clinical deterioration'
        ]
    elif risk_level == 'Moderate':
        recommendations = [
            'Regular monitoring recommended',
            'Repeat assessment in 4 hours',
            'Watch for changes in vital signs'
        ]
    else:
        recommendations = [
            'Continue standard monitoring',
            'Routine assessment intervals',
            'Document baseline parameters'
        ]
    
    # Add specific warnings
    if patient_data.get('Lactate', 0) > 2:
        recommendations.append('⚠️ Elevated lactate detected')
    if patient_data.get('SBP', 120) < 90:
        recommendations.append('⚠️ Hypotension - fluid resuscitation priority')
    
    return recommendations

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': True
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict sepsis risk"""
    try:
        data = request.get_json()
        
        # Map frontend field names to model feature names
        feature_mapping = {
            'age': 'age',
            'heart_rate': 'HR',
            'temperature': 'Temp',
            'respiratory_rate': 'Resp',
            'systolic_bp': 'SBP',
            'diastolic_bp': 'DBP',
            'wbc_count': 'WBC',
            'lactate': 'Lactate',
            'procalcitonin': 'Procalcitonin'
        }
        
        # Convert to model features
        patient_data = {}
        for frontend_name, model_name in feature_mapping.items():
            if frontend_name in data:
                patient_data[model_name] = data[frontend_name]
        
        # Create feature vector (in correct order)
        feature_order = ['age', 'HR', 'Temp', 'Resp', 'SBP', 'DBP', 'WBC', 'Lactate', 'Procalcitonin']
        features = [[patient_data.get(f, 0) for f in feature_order]]
        
        # Make prediction
        probability = model.predict_proba(features)[0][1]
        risk_level, risk_color = calculate_risk_level(probability)
        
        # Generate recommendations
        recommendations = generate_recommendations(risk_level, patient_data)
        
        result = {
            'probability': float(probability),
            'risk_level': risk_level,
            'risk_color': risk_color,
            'risk_score': int(probability * 100),
            'recommendations': recommendations
        }
        
        print(f"✅ Prediction: {risk_level} ({probability:.2%})")
        
        return jsonify(result)
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Get model information"""
    return jsonify({
        'model_type': 'XGBoost',
        'version': '1.0.0',
        'features': ['age', 'HR', 'Temp', 'Resp', 'SBP', 'DBP', 'WBC', 'Lactate', 'Procalcitonin']
    })

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 SEPSIS PREDICTION API SERVER")
    print("="*60)
    print("\n✅ API running at: http://localhost:5000")
    print("✅ Health check: http://localhost:5000/api/health")
    print("\nPress CTRL+C to stop\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)