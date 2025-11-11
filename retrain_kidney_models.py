"""
Script to retrain kidney disease prediction models with correct feature order.
Run this script to fix the feature order mismatch issue.
"""
import sys
import os

# Add models directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'models'))

from models.kidney_model import train_kidney_models

if __name__ == '__main__':
    print("=" * 60)
    print("Retraining Kidney Disease Prediction Models")
    print("=" * 60)
    print("\nThis will retrain the models with the correct feature order.")
    print("This should fix the issue where all predictions show high risk.\n")
    
    try:
        train_kidney_models()
        print("\n" + "=" * 60)
        print("SUCCESS: Models retrained successfully!")
        print("=" * 60)
        print("\nYou can now test the kidney prediction page with healthy values.")
        print("It should now correctly show 'Low risk' for healthy inputs.\n")
    except Exception as e:
        print("\n" + "=" * 60)
        print("ERROR: Failed to retrain models")
        print("=" * 60)
        print(f"\nError: {str(e)}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)

