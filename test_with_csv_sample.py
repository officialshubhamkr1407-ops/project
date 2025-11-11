"""Test with actual CSV sample to verify models"""
import pandas as pd
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from models.kidney_model import predict_kidney_disease

# Load CSV
df = pd.read_csv('models/kidney_disease.csv', na_values=['?', '\t?', ' ', '', '\t'])

# Strip whitespace
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].astype(str).str.strip()

# Map categorical
cat_map = {
    'rbc': {'normal': 0, 'abnormal': 1},
    'pc': {'normal': 0, 'abnormal': 1},
    'pcc': {'notpresent': 0, 'present': 1},
    'ba': {'notpresent': 0, 'present': 1},
    'htn': {'no': 0, 'yes': 1},
    'dm': {'no': 0, 'yes': 1},
    'cad': {'no': 0, 'yes': 1},
    'appet': {'good': 0, 'poor': 1},
    'pe': {'no': 0, 'yes': 1},
    'ane': {'no': 0, 'yes': 1},
}

for col, mapping in cat_map.items():
    if col in df.columns:
        df[col] = df[col].map(mapping)

# Fill missing
for col in df.columns:
    if col != 'classification':
        if df[col].dtype in [np.float64, np.int64]:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 0)

# Get healthy and disease samples
healthy_samples = df[df['classification'] == 'notckd'].head(3)
disease_samples = df[df['classification'] == 'ckd'].head(3)

feature_order = ['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba',
                 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc',
                 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']

print("Testing with ACTUAL CSV samples:")
print("=" * 60)

for idx, row in healthy_samples.iterrows():
    features = [row[f] for f in feature_order]
    result = predict_kidney_disease(features, force_reload=False)
    print(f"Healthy sample {idx}: Prediction={result} (Expected=0)")
    if result != 0:
        print(f"  ERROR: Should be 0!")
        print(f"  Features: age={row['age']}, bp={row['bp']}, sg={row['sg']}, al={row['al']}")

print("\n" + "=" * 60)

for idx, row in disease_samples.iterrows():
    features = [row[f] for f in feature_order]
    result = predict_kidney_disease(features, force_reload=False)
    print(f"Disease sample {idx}: Prediction={result} (Expected=1)")
    if result != 1:
        print(f"  WARNING: Should be 1!")

