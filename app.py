from flask import Flask, render_template, request, jsonify
import numpy as np
from models.kidney_model import predict_kidney_disease
from models.liver_model import predict_liver_disease

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/kidney')
def kidney():
    return render_template('kidney.html')

@app.route('/liver')
def liver():
    return render_template('liver.html')

@app.route('/predict_kidney', methods=['POST'])
def predict_kidney():
    try:
        # Get values from the form
        features = [
            float(request.form['age']),
            float(request.form['bp']),
            float(request.form['sg']),
            float(request.form['al']),
            float(request.form['su']),
            float(request.form['rbc']),
            float(request.form['pc']),
            float(request.form['pcc']),
            float(request.form['ba']),
            float(request.form['bgr']),
            float(request.form['bu']),
            float(request.form['sc']),
            float(request.form['sod']),
            float(request.form['pot']),
            float(request.form['hemo']),
            float(request.form['pcv']),
            float(request.form['wc']),
            float(request.form['rc']),
            float(request.form['htn']),
            float(request.form['dm']),
            float(request.form['cad']),
            float(request.form['appet']),
            float(request.form['pe']),
            float(request.form['ane'])
        ]

        # Make prediction (force reload to ensure latest models are used)
        prediction = predict_kidney_disease(features, force_reload=True)

        # Handle error case (-1)
        if prediction == -1:
            return jsonify({
                'success': False,
                'error': 'Failed to make prediction. Please check if models are loaded correctly.'
            })

        # Convert to native Python int for JSON serialization
        prediction = int(prediction)

        return jsonify({
            'success': True,
            'prediction': prediction,
            'message': 'High risk of kidney disease' if prediction == 1 else 'Low risk of kidney disease'
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })


@app.route('/predict_liver', methods=['POST'])
def predict_liver():
    try:
        # Get values from the form
        features = [
            float(request.form['age_of_patient']),
            float(request.form['gender_of_patient']),
            float(request.form['total_bilirubin']),
            float(request.form['direct_bilirubin']),
            float(request.form['alkphos_alkaline_phosphotase']),
            float(request.form['sgpt_alamine_aminotransferase']),
            float(request.form['sgot_aspartate_aminotransferase']),
            float(request.form['total_protiens']),
            float(request.form['alb_albumin']),
            float(request.form['ag_ratio_albumin_and_globulin_ratio'])
        ]

        # Make prediction
        prediction = predict_liver_disease(features)

        # Handle error case (-1)
        if prediction == -1:
            return jsonify({
                'success': False,
                'error': 'Failed to make prediction. Please check if models are loaded correctly.'
            })

        # Convert to native Python int for JSON serialization
        prediction = int(prediction)

        # Ensure message is correct (double-check to prevent any confusion)
        if prediction == 1:
            message = 'High risk of liver disease'
        elif prediction == 0:
            message = 'Low risk of liver disease'
        else:
            message = 'Unable to determine risk'
        
        return jsonify({
            'success': True,
            'prediction': prediction,
            'message': message
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })


if __name__ == '__main__':
    app.run(debug=True, port=5001)
