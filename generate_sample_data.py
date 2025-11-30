"""
Sample Data Generator for Sepsis Prediction
Creates synthetic patient data for testing the system
"""

import numpy as np
import pandas as pd
import os

def generate_sample_data(n_samples=5000, sepsis_ratio=0.15):
    """
    Generate synthetic patient data for sepsis prediction
    
    Args:
        n_samples: Number of patient records to generate
        sepsis_ratio: Ratio of sepsis cases (0.15 = 15%)
    
    Returns:
        DataFrame with patient data
    """
    np.random.seed(42)
    
    print(f"Generating {n_samples} patient records...")
    
    # Number of sepsis cases
    n_sepsis = int(n_samples * sepsis_ratio)
    n_normal = n_samples - n_sepsis
    
    # Generate normal patients (no sepsis)
    normal_data = {
        'patient_id': range(n_normal),
        'age': np.random.normal(55, 15, n_normal).clip(18, 95),
        'HR': np.random.normal(75, 12, n_normal).clip(50, 100),  # Heart Rate
        'Temp': np.random.normal(37.0, 0.3, n_normal).clip(36.0, 37.8),  # Temperature
        'Resp': np.random.normal(16, 3, n_normal).clip(12, 20),  # Respiratory Rate
        'SBP': np.random.normal(120, 15, n_normal).clip(100, 160),  # Systolic BP
        'DBP': np.random.normal(80, 10, n_normal).clip(60, 100),  # Diastolic BP
        'WBC': np.random.normal(7.5, 2, n_normal).clip(4.5, 11),  # White Blood Cells
        'Lactate': np.random.normal(1.2, 0.4, n_normal).clip(0.5, 2.0),
        'Procalcitonin': np.random.normal(0.1, 0.05, n_normal).clip(0.01, 0.4),
        'sepsis_label': 0
    }
    
    # Generate sepsis patients
    sepsis_data = {
        'patient_id': range(n_normal, n_samples),
        'age': np.random.normal(65, 12, n_sepsis).clip(30, 95),
        'HR': np.random.normal(105, 20, n_sepsis).clip(70, 160),  # Elevated
        'Temp': np.concatenate([
            np.random.normal(38.5, 0.8, n_sepsis//2).clip(38.0, 41.0),  # Fever
            np.random.normal(35.5, 0.5, n_sepsis//2).clip(34.0, 36.0)   # Hypothermia
        ]),
        'Resp': np.random.normal(24, 5, n_sepsis).clip(20, 40),  # Elevated
        'SBP': np.random.normal(90, 15, n_sepsis).clip(60, 120),  # Hypotension
        'DBP': np.random.normal(60, 12, n_sepsis).clip(40, 85),
        'WBC': np.concatenate([
            np.random.normal(16, 4, n_sepsis//2).clip(12, 30),  # Elevated
            np.random.normal(3, 0.8, n_sepsis//2).clip(1, 4)    # Decreased
        ]),
        'Lactate': np.random.normal(3.5, 1.5, n_sepsis).clip(2.0, 10.0),  # Elevated
        'Procalcitonin': np.random.normal(5.0, 3.0, n_sepsis).clip(0.5, 50.0),  # Elevated
        'sepsis_label': 1
    }
    
    # Combine datasets
    df_normal = pd.DataFrame(normal_data)
    df_sepsis = pd.DataFrame(sepsis_data)
    df = pd.concat([df_normal, df_sepsis], ignore_index=True)
    
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Add some missing values (realistic scenario)
    missing_rate = 0.05
    for col in ['WBC', 'Lactate', 'Procalcitonin']:
        missing_mask = np.random.random(len(df)) < missing_rate
        df.loc[missing_mask, col] = np.nan
    
    # Add timestamp (hourly measurements)
    df['timestamp'] = pd.date_range('2024-01-01', periods=len(df), freq='H')
    
    return df


def save_data(df, output_dir='data/raw'):
    """Save generated data to file"""
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, 'sample_training_data.csv')
    df.to_csv(output_path, index=False)
    
    print(f"\n✅ Data saved to: {output_path}")
    print(f"Total records: {len(df)}")
    print(f"Sepsis cases: {(df['sepsis_label'] == 1).sum()} ({(df['sepsis_label'] == 1).sum()/len(df)*100:.1f}%)")
    print(f"Normal cases: {(df['sepsis_label'] == 0).sum()} ({(df['sepsis_label'] == 0).sum()/len(df)*100:.1f}%)")
    
    return output_path


def display_sample_statistics(df):
    """Display statistics of the generated data"""
    print("\n" + "="*60)
    print("SAMPLE DATA STATISTICS")
    print("="*60)
    
    print("\nFeature Statistics:")
    print(df.describe().round(2))
    
    print("\n\nSepsis Distribution:")
    print(df['sepsis_label'].value_counts())
    
    print("\n\nMissing Values:")
    print(df.isnull().sum())
    
    print("\n\nSample Records (First 5):")
    print(df.head())


def main():
    """Generate sample data for testing"""
    print("="*60)
    print("SEPSIS SAMPLE DATA GENERATOR")
    print("="*60)
    
    # Generate data
    df = generate_sample_data(n_samples=5000, sepsis_ratio=0.15)
    
    # Display statistics
    display_sample_statistics(df)
    
    # Save to file
    output_path = save_data(df)
    
    print("\n" + "="*60)
    print("✅ Sample data generation complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Run: python src/data_preprocessing.py")
    print("2. Run: python src/model_training.py")
    print("3. Start API: python api/app.py")
    print("="*60)
    
    return output_path


if __name__ == "__main__":
    main()