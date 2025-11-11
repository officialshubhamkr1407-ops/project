"""
Test script to verify kidney prediction with healthy values.
This will help diagnose the feature order issue.
"""
import sys
import os

# Add models directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'models'))

from models.kidney_model import predict_kidney_disease, load_models
import joblib

def test_healthy_values():
    """Test prediction with healthy/low-risk values"""
    print("=" * 60)
    print("Testing Kidney Prediction with Healthy Values")
    print("=" * 60)
    
    # Load models first
    print("\nLoading models...")
    scaler = load_models()
    if scaler is None:
        print("ERROR: Failed to load models!")
        return
    
    # Healthy values (low risk) - matching the form's low risk sample
    healthy_features = [
        35,      # age
        120,     # bp
        1.015,   # sg
        0,       # al
        0,       # su
        0,       # rbc (Normal)
        0,       # pc (Normal)
        0,       # pcc (Not Present)
        0,       # ba (Not Present)
        100,     # bgr
        30,      # bu
        0.9,     # sc
        140,     # sod
        4.5,     # pot
        15,      # hemo
        45,      # pcv
        7000,    # wc
        4.5,     # rc
        0,       # htn (No)
        0,       # dm (No)
        0,       # cad (No)
        0,       # appet (Good)
        0,       # pe (No)
        0        # ane (No)
    ]
    
    print(f"\nTesting with {len(healthy_features)} healthy features...")
    print(f"Features: {healthy_features[:5]}... (showing first 5)")
    
    # Make prediction
    prediction = predict_kidney_disease(healthy_features)
    
    print("\n" + "=" * 60)
    print("PREDICTION RESULT")
    print("=" * 60)
    print(f"Prediction: {prediction}")
    if prediction == 0:
        print("✓ CORRECT: Low risk (as expected for healthy values)")
    elif prediction == 1:
        print("✗ ERROR: High risk (unexpected for healthy values!)")
        print("\nThis indicates a problem with the model or feature order.")
    else:
        print(f"✗ ERROR: Invalid prediction result ({prediction})")
    
    print("=" * 60)
    
    # Also test high risk values
    print("\n" + "=" * 60)
    print("Testing with High Risk Values (for comparison)")
    print("=" * 60)
    
    high_risk_features = [
        65,      # age
        180,     # bp
        1.010,   # sg
        3,       # al
        2,       # su
        1,       # rbc (Abnormal)
        1,       # pc (Abnormal)
        1,       # pcc (Present)
        1,       # ba (Present)
        250,     # bgr
        100,     # bu
        3.5,     # sc
        130,     # sod
        5.5,     # pot
        10,      # hemo
        30,      # pcv
        12000,   # wc
        3.0,     # rc
        1,       # htn (Yes)
        1,       # dm (Yes)
        1,       # cad (Yes)
        1,       # appet (Poor)
        1,       # pe (Yes)
        1        # ane (Yes)
    ]
    
    prediction_high = predict_kidney_disease(high_risk_features)
    print(f"Prediction: {prediction_high}")
    if prediction_high == 1:
        print("✓ CORRECT: High risk (as expected for high-risk values)")
    elif prediction_high == 0:
        print("⚠ WARNING: Low risk (unexpected for high-risk values)")
    else:
        print(f"✗ ERROR: Invalid prediction result ({prediction_high})")
    
    print("=" * 60)

if __name__ == '__main__':
    test_healthy_values()

