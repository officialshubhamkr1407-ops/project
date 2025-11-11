"""Quick test to verify kidney prediction is working"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from models.kidney_model import predict_kidney_disease

# Test with healthy values
healthy = [35, 120, 1.015, 0, 0, 0, 0, 0, 0, 100, 30, 0.9, 140, 4.5, 15, 45, 7000, 4.5, 0, 0, 0, 0, 0, 0]
print("Testing with healthy values...")
result = predict_kidney_disease(healthy, force_reload=True)
print(f"Result: {result} (0=Low risk, 1=High risk)")
print(f"Expected: 0 (Low risk)")
print(f"✓ CORRECT" if result == 0 else "✗ ERROR: Should be 0 (Low risk)")

