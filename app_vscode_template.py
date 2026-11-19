
from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the pre-trained Random Forest model and scaler
try:
    model = joblib.load('best_random_forest_model.pkl')
    scaler = joblib.load('scaler.pkl')
    print("Model and Scaler loaded successfully!")
except Exception as e:
    print(f"Error loading serialized assets: {e}")

@app.route('/')
def home():
    # Route 1: Home Page showing project overview
    return render_template('home.html')

@app.route('/predict_form')
def predict_form():
    # Route 2: Render the prediction input form
    return render_template('prediction.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Route 3: Handle prediction request and display results
    try:
        # Retrieve features from form inputs
        input_data = {
            'employees': float(request.form['employees']),
            'net_income': float(request.form['net_income']),
            'market_cap': float(request.form['market_cap']),
            'legal_units_count': float(request.form['legal_units_count']),
            'direct_subsidiaries_count': float(request.form['direct_subsidiaries_count']),
            'max_hierarchy_depth': float(request.form['max_hierarchy_depth']),
            'enterprise_group_jurisdiction': float(request.form['enterprise_group_jurisdiction']),
            'company_type': float(request.form['company_type']),
            'is_standalone_entity_True': float(request.form['is_standalone_entity'])
        }

        # Standardize features matching model columns order
        feature_cols = ['employees', 'net_income', 'market_cap', 'legal_units_count', 
                        'direct_subsidiaries_count', 'max_hierarchy_depth', 
                        'enterprise_group_jurisdiction', 'company_type', 'is_standalone_entity_True']
        
        features_df = pd.DataFrame([input_data])[feature_cols]
        scaled_features = scaler.transform(features_df)
        
        # Execute Random Forest prediction
        raw_prediction = model.predict(scaled_features)[0]
        formatted_prediction = f"${raw_prediction:,.2f}"
        
        return render_template('result.html', 
                               prediction=formatted_prediction,
                               inputs=request.form)
    except Exception as e:
        return render_template('result.html', error_msg=str(e))

if __name__ == '__main__':
    app.run(debug=True, port=5000)
