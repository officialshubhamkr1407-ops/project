import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import joblib
import os

# Initialize the models
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42)

def load_models():
    """Load the trained models and scaler"""
    global rf_model, xgb_model
    model_dir = "models"   # <-- fixed for Colab
    try:
        # Load Random Forest model
        rf_path = os.path.join(model_dir, 'kidney_rf_model.pkl')
        if os.path.exists(rf_path):
            rf_model = joblib.load(rf_path)

        # Load XGBoost model
        xgb_path = os.path.join(model_dir, 'kidney_xgb_model.pkl')
        if os.path.exists(xgb_path):
            xgb_model = joblib.load(xgb_path)

        # Load scaler
        scaler_path = os.path.join(model_dir, 'kidney_scaler.pkl')
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            return scaler
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        return None


def predict_kidney_disease(features):
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

    Returns:
    int: 1 if high risk, 0 if low risk
    """
    try:
        # Convert features to numpy array and reshape
        features_array = np.array(features, dtype=float).reshape(1, -1)

        # Load scaler
        scaler = load_models()
        if scaler is not None:
            features_array = scaler.transform(features_array)

        # Make predictions using both models
        rf_pred = rf_model.predict(features_array)[0]
        xgb_pred = xgb_model.predict(features_array)[0]

        # Ensemble prediction (majority voting)
        final_prediction = int((rf_pred + xgb_pred) >= 1)
        return final_prediction

    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        # Return -1 to indicate error
        return -1


# Load models when the module is imported
load_models()

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
    X = df.drop('classification', axis=1)
    y = df['classification']

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
