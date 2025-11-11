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
        rf_path = os.path.join(model_dir, 'liver_rf_model.pkl')
        xgb_path = os.path.join(model_dir, 'liver_xgb_model.pkl')
        scaler_path = os.path.join(model_dir, 'liver_scaler.pkl')
        
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
        
        # Verify feature count matches expected (10 features)
        if len(rf_model.feature_importances_) != 10:
            print(f"ERROR: Random Forest model expects {len(rf_model.feature_importances_)} features, but we need 10!")
            return None
        if len(xgb_model.feature_importances_) != 10:
            print(f"ERROR: XGBoost model expects {len(xgb_model.feature_importances_)} features, but we need 10!")
            return None
        
        models_loaded = True
        print("Liver models loaded and validated successfully")
        return scaler
            
    except Exception as e:
        print(f"ERROR loading liver models: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def predict_liver_disease(features):
    """
    Predict liver disease risk based on input features.

    Parameters:
    features (list): List of 10 features in the following order:
        1. Age
        2. Gender (1 for male, 0 for female)
        3. Total Bilirubin
        4. Direct Bilirubin
        5. Alkaline Phosphatase
        6. Alamine Aminotransferase
        7. Aspartate Aminotransferase
        8. Total Proteins
        9. Albumin
        10. Albumin and Globulin Ratio

    Returns:
    int: 1 if high risk, 0 if low risk, -1 if error
    """
    global scaler
    try:
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
        if len(features) != 10:
            print(f"ERROR: Expected 10 features, got {len(features)}")
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

        # Ensemble prediction (majority voting: both models must agree for high risk)
        # If both predict 1, final is 1 (high risk)
        # If both predict 0, final is 0 (low risk)
        # If they disagree, use the average (more conservative approach)
        if rf_pred == xgb_pred:
            final_prediction = rf_pred
        else:
            # Models disagree - use average (0.5 rounds to 0, more conservative)
            # This prevents false positives when models disagree
            final_prediction = int((rf_pred + xgb_pred) / 2)
        
        # Convert to native Python int (not numpy int64) for JSON serialization
        final_prediction = int(final_prediction)
        
        # Debug output (can be removed in production)
        print(f"DEBUG: RF prediction={rf_pred}, XGB prediction={xgb_pred}, Final={final_prediction}")
        
        return final_prediction

    except Exception as e:
        print(f"Error in liver disease prediction: {str(e)}")
        # Return -1 to indicate error
        return -1


# Load models when the module is imported
scaler = load_models()

# ------------------- Training Function -------------------
def train_liver_models():
    """
    Train and save scaler, RandomForest, and XGBoost models using indian_liver_patient.csv
    """

    data_path = os.path.join(os.path.dirname(__file__), 'Liver Patient Dataset (LPD)_train.csv')
    df = pd.read_csv(data_path, encoding='latin1')
    df.columns = df.columns.str.strip()

    # Map Gender to numeric: Male=1, Female=0
    if 'Gender of the patient' in df.columns:
        df['Gender of the patient'] = df['Gender of the patient'].map({'Male': 1, 'Female': 0})

    # Remove rows with missing values
    df = df.dropna()

    # Features and target (use exact column names from CSV)
    X = df[['Age of the patient', 'Gender of the patient', 'Total Bilirubin', 'Direct Bilirubin',
            'Alkphos Alkaline Phosphotase', 'Sgpt Alamine Aminotransferase',
            'Sgot Aspartate Aminotransferase', 'Total Protiens', 'ALB Albumin',
            'A/G Ratio Albumin and Globulin Ratio']]
    # 1 = liver disease, 2 = no disease; map 2 to 0
    y = df['Result'].apply(lambda v: 1 if v == 1 else 0)

    # Standardize features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply SMOTE for class balancing
    try:
        from imblearn.over_sampling import SMOTE
        sm = SMOTE(random_state=42)
        X_res, y_res = sm.fit_resample(X_scaled, y)
        print(f"After SMOTE: {sum(y_res==0)} healthy, {sum(y_res==1)} disease")
    except ImportError:
        print("imblearn not installed, skipping SMOTE.")
        X_res, y_res = X_scaled, y

    # Train models
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    xgbc = xgb.XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
    rf.fit(X_res, y_res)
    xgbc.fit(X_res, y_res)

    # Model evaluation
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    y_pred = rf.predict(X_scaled)
    print("RandomForest on original data:")
    print("Accuracy:", accuracy_score(y, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y, y_pred))
    print(classification_report(y, y_pred))

    # Save models and scaler
    joblib.dump(scaler, os.path.join(os.path.dirname(__file__), 'liver_scaler.pkl'))
    joblib.dump(rf, os.path.join(os.path.dirname(__file__), 'liver_rf_model.pkl'))
    joblib.dump(xgbc, os.path.join(os.path.dirname(__file__), 'liver_xgb_model.pkl'))
    print('Liver model training complete. Models and scaler saved.')
