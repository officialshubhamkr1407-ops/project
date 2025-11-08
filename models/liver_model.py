import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import joblib
import os

# --- Globals to hold loaded models and scaler ---
rf_model = None
xgb_model = None
scaler = None

def load_models():
    """Load the trained models and scaler into global variables."""
    global rf_model, xgb_model, scaler
    
    # Get the directory where this file (liver_model.py) is located
    model_dir = os.path.dirname(__file__) 
    
    try:
        rf_path = os.path.join(model_dir, 'liver_rf_model.pkl')
        if os.path.exists(rf_path):
            rf_model = joblib.load(rf_path)
            print("Loaded liver_rf_model.pkl")

        xgb_path = os.path.join(model_dir, 'liver_xgb_model.pkl')
        if os.path.exists(xgb_path):
            xgb_model = joblib.load(xgb_path)
            print("Loaded liver_xgb_model.pkl")

        scaler_path = os.path.join(model_dir, 'liver_scaler.pkl')
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            print("Loaded liver_scaler.pkl")
        
        if rf_model is None or xgb_model is None or scaler is None:
            print("Warning: One or more liver model/scaler files were not found.")
            print("Please run 'python models/liver_model.py' to train and create them.")

    except Exception as e:
        print(f"Error loading liver models: {str(e)}")


def predict_liver_disease(features):
    """
    Predict liver disease risk based on input features.

    Parameters:
    features (list): List of 10 features...
    """
    global rf_model, xgb_model, scaler # Use the globally loaded models

    # Check if models are loaded. If not, training hasn't been run.
    if rf_model is None or xgb_model is None or scaler is None:
        print("Error: Liver models are not loaded. Cannot predict.")
        return -1 # Return error code
    
    try:
        # Convert features to numpy array and reshape
        features_array = np.array(features, dtype=float).reshape(1, -1)

        # Scale features
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


# --- Load models ONCE when the module is imported by app.py ---
load_models() 

# ------------------- Training Function -------------------
def train_liver_models():
    """
    Train and save scaler, RandomForest, and XGBoost models using indian_liver_patient.csv
    """
    
    # Define paths relative to this file's location
    model_dir = os.path.dirname(__file__)
    data_path = os.path.join(model_dir, 'Liver Patient Dataset (LPD)_train.csv')

    try:
        df = pd.read_csv(data_path, encoding='latin1')
    except FileNotFoundError:
        print(f"Error: Training data file not found at {data_path}")
        print("Please make sure 'Liver Patient Dataset (LPD)_train.csv' is in the 'models' directory.")
        return

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
    print("--- RandomForest Evaluation on original data ---")
    print("Accuracy:", accuracy_score(y, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y, y_pred))
    print(classification_report(y, y_pred))

    # Save models and scaler
    joblib.dump(scaler, os.path.join(model_dir, 'liver_scaler.pkl'))
    joblib.dump(rf, os.path.join(model_dir, 'liver_rf_model.pkl'))
    joblib.dump(xgbc, os.path.join(model_dir, 'liver_xgb_model.pkl'))
    print('Liver model training complete. Models and scaler saved in models/ directory.')


# ------------------- Make Script Runnable -------------------
# This block will only run when you execute this file directly
# It will NOT run when app.py imports it
if __name__ == "__main__":
    print("Starting liver model training...")
    train_liver_models()