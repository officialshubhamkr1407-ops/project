# PredictWell: AI-Powered Disease Risk Prediction

PredictWell is a web-based healthcare assistant designed to predict the risk of Liver and Kidney disease. It provides an intuitive, user-friendly interface for entering medical parameters and delivers real-time risk assessments using machine learning models.

This application is built with a **Flask** backend, which serves the machine learning models (a **RandomForest** and **XGBoost** ensemble), and a **Bootstrap/jQuery** frontend to capture user input and display results.

## 🚀 Features

  * **Liver Disease Prediction:** Assesses risk based on 10 key medical features, including Bilirubin levels, Albumin, and various enzymes.
  * **Kidney Disease Prediction:** Assesses risk based on 24 parameters like blood pressure, specific gravity, albumin, and blood cell counts.
  * **Interactive UI:** Uses responsive sliders and dropdowns for easy data entry.
  * **Real-time Feedback:** Provides instant risk analysis (High Risk / Low Risk) with a clear color-coded alert (Red / Green).
  * **Asynchronous Processing:** Features a loading spinner on the "Predict" button to provide user feedback while the model processes the request.

## 💻 Technology Stack

  * **Backend:** Python, Flask
  * **Machine Learning:** Scikit-learn, XGBoost, Pandas, NumPy, Joblib, Imbalanced-learn (SMOTE)
  * **Frontend:** HTML5, CSS3, Bootstrap 5, JavaScript, jQuery


## ⚠️ Disclaimer

This is a demo project for educational and technical demonstration purposes. The predictions made by this tool are **not a substitute for professional medical advice, diagnosis, or treatment.** Always consult a qualified healthcare provider with any questions you may have regarding a medical condition.
