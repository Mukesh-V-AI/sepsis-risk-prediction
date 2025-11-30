"""
Simple model training script for sepsis prediction
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
import xgboost as xgb
import joblib
import os
from datetime import datetime

print("="*60)
print("MODEL TRAINING")
print("="*60)

# Load processed data
print("\n1. Loading processed data...")
X_train = pd.read_csv('data/processed/X_train.csv')
X_test = pd.read_csv('data/processed/X_test.csv')
y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()
y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()

print(f"   Training set: {X_train.shape}")
print(f"   Test set: {X_test.shape}")

# Create models directory
os.makedirs('models/trained_models', exist_ok=True)

# ========== RANDOM FOREST ==========
print("\n" + "="*60)
print("TRAINING RANDOM FOREST")
print("="*60)

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)

print("Training...")
rf_model.fit(X_train, y_train)
print("✅ Training complete!")

print("\nEvaluating on test set...")
rf_pred = rf_model.predict(X_test)
rf_proba = rf_model.predict_proba(X_test)[:, 1]

rf_accuracy = accuracy_score(y_test, rf_pred)
rf_precision = precision_score(y_test, rf_pred)
rf_recall = recall_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred)
rf_auc = roc_auc_score(y_test, rf_proba)

print(f"\n📊 Random Forest Results:")
print(f"   Accuracy:  {rf_accuracy:.4f}")
print(f"   Precision: {rf_precision:.4f}")
print(f"   Recall:    {rf_recall:.4f}")
print(f"   F1-Score:  {rf_f1:.4f}")
print(f"   AUC-ROC:   {rf_auc:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, rf_pred))

# Save model
rf_path = 'models/trained_models/random_forest_model.pkl'
joblib.dump(rf_model, rf_path)
print(f"\n✅ Model saved to: {rf_path}")

# ========== XGBOOST ==========
print("\n" + "="*60)
print("TRAINING XGBOOST")
print("="*60)

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

xgb_model = xgb.XGBClassifier(
    max_depth=6,
    learning_rate=0.1,
    n_estimators=100,
    objective='binary:logistic',
    random_state=42,
    scale_pos_weight=scale_pos_weight,
    eval_metric='auc',
    use_label_encoder=False
)

print("Training...")
xgb_model.fit(X_train, y_train)
print("✅ Training complete!")

print("\nEvaluating on test set...")
xgb_pred = xgb_model.predict(X_test)
xgb_proba = xgb_model.predict_proba(X_test)[:, 1]

xgb_accuracy = accuracy_score(y_test, xgb_pred)
xgb_precision = precision_score(y_test, xgb_pred)
xgb_recall = recall_score(y_test, xgb_pred)
xgb_f1 = f1_score(y_test, xgb_pred)
xgb_auc = roc_auc_score(y_test, xgb_proba)

print(f"\n📊 XGBoost Results:")
print(f"   Accuracy:  {xgb_accuracy:.4f}")
print(f"   Precision: {xgb_precision:.4f}")
print(f"   Recall:    {xgb_recall:.4f}")
print(f"   F1-Score:  {xgb_f1:.4f}")
print(f"   AUC-ROC:   {xgb_auc:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, xgb_pred))

# Save model
xgb_path = 'models/trained_models/xgboost_model.pkl'
joblib.dump(xgb_model, xgb_path)
print(f"\n✅ Model saved to: {xgb_path}")

# ========== COMPARISON ==========
print("\n" + "="*60)
print("MODEL COMPARISON")
print("="*60)

comparison = pd.DataFrame({
    'Model': ['Random Forest', 'XGBoost'],
    'Accuracy': [rf_accuracy, xgb_accuracy],
    'Precision': [rf_precision, xgb_precision],
    'Recall': [rf_recall, xgb_recall],
    'F1-Score': [rf_f1, xgb_f1],
    'AUC-ROC': [rf_auc, xgb_auc]
})

print(comparison.to_string(index=False))

# Determine best model
if xgb_auc >= rf_auc:
    best_model = 'XGBoost'
    best_auc = xgb_auc
else:
    best_model = 'Random Forest'
    best_auc = rf_auc

print(f"\n🏆 BEST MODEL: {best_model} (AUC-ROC: {best_auc:.4f})")

# Feature importance (XGBoost)
print("\n" + "="*60)
print("TOP 10 IMPORTANT FEATURES (XGBoost)")
print("="*60)

feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance.head(10).to_string(index=False))

print("\n" + "="*60)
print("✅ TRAINING COMPLETE!")
print("="*60)
print(f"\nModels saved:")
print(f"  • {rf_path}")
print(f"  • {xgb_path}")
print(f"\nNext step: python api/app.py")