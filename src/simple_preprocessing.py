"""
Simple preprocessing script for sample sepsis data
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

print("="*60)
print("DATA PREPROCESSING")
print("="*60)

# Load sample data
print("\n1. Loading data...")
df = pd.read_csv('data/raw/sample_training_data.csv')
print(f"   Loaded {len(df)} records")

# Separate features and target
print("\n2. Separating features and target...")
X = df.drop(['timestamp', 'patient_id', 'sepsis_label'], axis=1)
y = df['sepsis_label']
print(f"   Features: {X.shape}")
print(f"   Target distribution:")
print(f"   - No Sepsis: {(y==0).sum()} ({(y==0).sum()/len(y)*100:.1f}%)")
print(f"   - Sepsis: {(y==1).sum()} ({(y==1).sum()/len(y)*100:.1f}%)")

# Handle missing values
print("\n3. Handling missing values...")
missing = X.isnull().sum()
if missing.sum() > 0:
    print(f"   Missing values found:")
    print(missing[missing > 0])
    X = X.fillna(X.median())
    print("   ✅ Filled with median values")
else:
    print("   No missing values found")

# Split data
print("\n4. Splitting data (80% train, 20% test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   Training set: {X_train.shape}")
print(f"   Test set: {X_test.shape}")

# Save processed data
print("\n5. Saving processed data...")
os.makedirs('data/processed', exist_ok=True)

X_train.to_csv('data/processed/X_train.csv', index=False)
X_test.to_csv('data/processed/X_test.csv', index=False)
y_train.to_csv('data/processed/y_train.csv', index=False, header=['sepsis_label'])
y_test.to_csv('data/processed/y_test.csv', index=False, header=['sepsis_label'])

print("   ✅ Saved to data/processed/")
print("      - X_train.csv")
print("      - X_test.csv")
print("      - y_train.csv")
print("      - y_test.csv")

print("\n" + "="*60)
print("✅ PREPROCESSING COMPLETE!")
print("="*60)
print("\nNext step: python src/simple_training.py")