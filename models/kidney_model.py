import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import joblib
import os

# Initialize the models
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42)
scaler = None  # Global scaler variable

def load_models():
    """Load the trained models and scaler"""
    global rf_model, xgb_model, scaler
    model_dir = os.path.dirname(__file__)  # Use the directory where this file is located
    models_loaded = False
    scaler = None
    
    try:
        # Load Random Forest model
        rf_path = os.path.join(model_dir, 'kidney_rf_model.pkl')
        xgb_path = os.path.join(model_dir, 'kidney_xgb_model.pkl')
        scaler_path = os.path.join(model_dir, 'kidney_scaler.pkl')
        
        # Check if all required files exist
        if not os.path.exists(rf_path):
            print(f"ERROR: Random Forest model not found at {rf_path}")
            return None
        if not os.path.exists(xgb_path):
            print(f"ERROR: XGBoost model not found at {xgb_path}")
            return None
        if not os.path.exists(scaler_path):
            print(f"ERROR: Scaler not found at {scaler_path}")
            return None
        
        # All files exist, load them
        rf_model = joblib.load(rf_path)
        xgb_model = joblib.load(xgb_path)
        scaler = joblib.load(scaler_path)
        
        # Verify models are trained (have feature_importances_ attribute)
        if not hasattr(rf_model, 'feature_importances_'):
            print("ERROR: Random Forest model appears to be untrained!")
            return None
        if not hasattr(xgb_model, 'feature_importances_'):
            print("ERROR: XGBoost model appears to be untrained!")
            return None
        
        # Verify feature count matches expected (24 features)
        if len(rf_model.feature_importances_) != 24:
            print(f"ERROR: Random Forest model expects {len(rf_model.feature_importances_)} features, but we need 24!")
            return None
        if len(xgb_model.feature_importances_) != 24:
            print(f"ERROR: XGBoost model expects {len(xgb_model.feature_importances_)} features, but we need 24!")
            return None
        
        models_loaded = True
        print("Kidney models loaded and validated successfully")
        return scaler
            
    except Exception as e:
        print(f"ERROR loading kidney models: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def predict_kidney_disease(features, force_reload=False):
    """
    Predict kidney disease risk based on input features

    Parameters:
    features (list): List of 24 features in the following order:
    1. Age
    2. Blood Pressure
    3. Specific Gravity
    4. Albumin
    5. Sugar
    6. Red Blood Cells
    7. Pus Cell
    8. Pus Cell Clumps
    9. Bacteria
    10. Blood Glucose Random
    11. Blood Urea
    12. Serum Creatinine
    13. Sodium
    14. Potassium
    15. Hemoglobin
    16. Packed Cell Volume
    17. White Blood Cell Count
    18. Red Blood Cell Count
    19. Hypertension
    20. Diabetes Mellitus
    21. Coronary Artery Disease
    22. Appetite
    23. Pedal Edema
    24. Anemia
    force_reload (bool): Force reload models from disk (useful after retraining)

    Returns:
    int: 1 if high risk, 0 if low risk, -1 if error
    """
    global scaler, rf_model, xgb_model
    try:
        # Force reload models if requested (useful after retraining)
        if force_reload:
            print("Force reloading models...")
            scaler = None
            rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
            xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42)
        
        # Check if models are loaded (check if scaler is available)
        if scaler is None:
            print("Warning: Models not loaded, attempting to load now...")
            scaler = load_models()
            if scaler is None:
                print("ERROR: Cannot make prediction - models not loaded!")
                return -1
        
        # Convert features to numpy array and reshape
        features_array = np.array(features, dtype=float).reshape(1, -1)
        
        # Validate feature count
        if len(features) != 24:
            print(f"ERROR: Expected 24 features, got {len(features)}")
            return -1

        # Transform features using scaler
        features_array = scaler.transform(features_array)

        # Verify models are trained before making predictions
        if not hasattr(rf_model, 'feature_importances_') or not hasattr(xgb_model, 'feature_importances_'):
            print("ERROR: Models are not trained! Cannot make prediction.")
            return -1
        
        # Make predictions using both models
        rf_pred = rf_model.predict(features_array)[0]
        xgb_pred = xgb_model.predict(features_array)[0]
        
        # Get prediction probabilities for better debugging
        rf_proba = rf_model.predict_proba(features_array)[0]
        xgb_proba = xgb_model.predict_proba(features_array)[0]

        # Ensemble prediction (conservative approach: require both models to agree for high risk)
        # If both predict 1, final is 1 (high risk)
        # If both predict 0, final is 0 (low risk)
        # If they disagree, use the more conservative prediction (low risk/0)
        # This prevents false positives - only predict high risk if both models agree
        if rf_pred == xgb_pred:
            final_prediction = rf_pred
        else:
            # Models disagree - use conservative approach: predict low risk (0)
            # This prevents false positives when models disagree
            final_prediction = 0
        
        # Convert to native Python int (not numpy int64) for JSON serialization
        final_prediction = int(final_prediction)
        
        # Debug output
        print(f"DEBUG: RF prediction={rf_pred} (proba: {rf_proba}), XGB prediction={xgb_pred} (proba: {xgb_proba}), Final={final_prediction}")
        print(f"DEBUG: Input features (first 5): {features[:5]}")
        
        return final_prediction

    except Exception as e:
        print(f"Error in kidney disease prediction: {str(e)}")
        # Return -1 to indicate error
        return -1


# Load models when the module is imported
scaler = load_models()

# ------------------- Training Function -------------------
def train_kidney_models():
    """
    Train and save scaler, RandomForest, and XGBoost models using kidney_disease.csv
    """
    data_path = os.path.join(os.path.dirname(__file__), 'kidney_disease.csv')

    # Read CSV, treat '?' and whitespace as NaN
    df = pd.read_csv(data_path, na_values=['?', '\t?', ' ', '', '\t'])

    # Drop id column if present
    if 'id' in df.columns:
        df = df.drop('id', axis=1)

    # Strip whitespace from string columns
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype(str).str.strip()

    # Map categorical columns to numeric
    cat_map = {
        'rbc': {'normal': 0, 'abnormal': 1},
        'pc': {'normal': 0, 'abnormal': 1},
        'pcc': {'notpresent': 0, 'present': 1},
        'ba': {'notpresent': 0, 'present': 1},
        'htn': {'no': 0, 'yes': 1},
        'dm': {'no': 0, 'yes': 1, ' yes': 1, ' yes ': 1, 'no ': 0},
        'cad': {'no': 0, 'yes': 1},
        'appet': {'good': 0, 'poor': 1},
        'pe': {'no': 0, 'yes': 1},
        'ane': {'no': 0, 'yes': 1},
        'classification': {'ckd': 1, 'notckd': 0, 'ckd': 1, 'ckd\t': 1}
    }
    for col, mapping in cat_map.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)

    # Fill missing values with median for numeric, mode for categorical
    for col in df.columns:
        if df[col].dtype in [np.float64, np.int64]:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])

    # Features and target
    # IMPORTANT: Order features to match the prediction order exactly
    # This order must match the order in app.py predict_kidney() function
    expected_feature_order = [
        'age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba',
        'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc',
        'htn', 'dm', 'cad', 'appet', 'pe', 'ane'
    ]
    
    # Get all feature columns (exclude classification)
    feature_cols = [col for col in df.columns if col != 'classification']
    
    # Check if all expected features exist
    missing_features = [f for f in expected_feature_order if f not in feature_cols]
    if missing_features:
        print(f"WARNING: Missing features in CSV: {missing_features}")
        print(f"Available features: {feature_cols}")
    
    # Reorder features to match expected order (only include features that exist)
    ordered_features = [f for f in expected_feature_order if f in feature_cols]
    
    # If there are extra features in CSV, add them at the end (shouldn't happen, but handle it)
    extra_features = [f for f in feature_cols if f not in expected_feature_order]
    if extra_features:
        print(f"WARNING: Extra features in CSV (will be ignored): {extra_features}")
    
    X = df[ordered_features]
    y = df['classification']
    
    print(f"Training with {len(ordered_features)} features in order: {ordered_features}")

    # Standardize features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train models
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    xgbc = xgb.XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
    rf.fit(X_scaled, y)
    xgbc.fit(X_scaled, y)

    # Save models and scaler
    joblib.dump(scaler, os.path.join(os.path.dirname(__file__), 'kidney_scaler.pkl'))
    joblib.dump(rf, os.path.join(os.path.dirname(__file__), 'kidney_rf_model.pkl'))
    joblib.dump(xgbc, os.path.join(os.path.dirname(__file__), 'kidney_xgb_model.pkl'))
    print('Training complete. Models and scaler saved.')
