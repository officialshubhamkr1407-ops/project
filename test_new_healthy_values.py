"""Test with corrected healthy values"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from models.kidney_model import predict_kidney_disease

# Updated healthy values based on actual healthy ranges from CSV
# bp: 70 (healthy range 60-80), sg: 1.02 (all healthy samples), rc: 5.4 (mean 5.38)
healthy_features = [
    40,      # age
    70,      # bp (was 120 - too high!)
    1.02,    # sg (was 1.015 - too low!)
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
    4.3,     # pot
    15,      # hemo
    46,      # pcv
    7000,    # wc
    5.4,     # rc (was 4.5 - too low!)
    0,       # htn (No)
    0,       # dm (No)
    0,       # cad (No)
    0,       # appet (Good)
    0,       # pe (No)
    0        # ane (No)
]

print("Testing with CORRECTED healthy values...")
print("=" * 60)
print("Key changes:")
print("  bp: 120 -> 70 (healthy range: 60-80)")
print("  sg: 1.015 -> 1.02 (healthy samples: 1.02)")
print("  rc: 4.5 -> 5.4 (healthy mean: 5.38)")
print("=" * 60)

result = predict_kidney_disease(healthy_features, force_reload=True)
print(f"\nResult: {result} (0=Low risk, 1=High risk)")
print(f"Expected: 0 (Low risk)")
if result == 0:
    print("SUCCESS: Correctly predicts Low risk!")
else:
    print("ERROR: Still predicting High risk!")

