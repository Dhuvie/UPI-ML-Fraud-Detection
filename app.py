"""
Flask web application for UPI fraud detection
"""

from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Model paths
MODEL_PATH = 'models/xgboost.pkl'
SCALER_PATH = 'models/scaler.pkl'
ENCODERS_PATH = 'models/label_encoders.pkl'

model = None
scaler = None
label_encoders = None


def load_models():
    """Load trained model and preprocessing objects"""
    global model, scaler, label_encoders
    
    if not os.path.exists(MODEL_PATH):
        print("Error: Models not found. Run 'python main.py' first.")
        return False
    
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    label_encoders = joblib.load(ENCODERS_PATH)
    print("Models loaded successfully")
    return True


def preprocess_input(data):
    """Preprocess user input for prediction"""
    # Create dataframe
    df = pd.DataFrame([data])
    
    # Encode categorical features FIRST (before feature engineering)
    for col, encoder in label_encoders.items():
        if col in df.columns:
            try:
                df[col] = encoder.transform(df[col].astype(str))
            except:
                df[col] = 0
    
    # Engineer features (same order as training)
    if 'step' in df.columns:
        df['hour'] = df['step'] % 24
        df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
    
    if 'amount' in df.columns:
        df['amount_log'] = np.log1p(df['amount'])
        df['is_round_amount'] = (df['amount'] % 1000 == 0).astype(int)
    
    if 'oldbalanceOrg' in df.columns and 'newbalanceOrig' in df.columns:
        df['balance_change_orig'] = df['newbalanceOrig'] - df['oldbalanceOrg']
    
    if 'oldbalanceDest' in df.columns and 'newbalanceDest' in df.columns:
        df['balance_change_dest'] = df['newbalanceDest'] - df['oldbalanceDest']
    
    # Get feature names from training data
    X_train_sample = pd.read_csv('data/processed/X_train.csv')
    expected_features = X_train_sample.columns.tolist()
    
    # Ensure all expected features exist
    for feature in expected_features:
        if feature not in df.columns:
            df[feature] = 0
    
    # Select only the features used in training (in correct order)
    df = df[expected_features]
    
    # Scale features
    X = scaler.transform(df.values)
    
    return X


@app.route('/')
def home():
    """Render home page"""
    return render_template('index.html')


def validate_transaction_data(data):
    """Validate transaction data"""
    required_fields = ['step', 'type', 'amount', 'nameOrig', 'oldbalanceOrg', 
                      'newbalanceOrig', 'nameDest', 'oldbalanceDest', 
                      'newbalanceDest', 'isFlaggedFraud']
    
    for field in required_fields:
        if field not in data:
            raise ValueError(f"Missing required field: {field}")
    
    # Validate numeric fields
    if data['amount'] < 0:
        raise ValueError("Amount cannot be negative")
    if data['oldbalanceOrg'] < 0 or data['newbalanceOrig'] < 0:
        raise ValueError("Sender balances cannot be negative")
    if data['oldbalanceDest'] < 0 or data['newbalanceDest'] < 0:
        raise ValueError("Receiver balances cannot be negative")
    
    # Validate transaction type
    valid_types = ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN']
    if data['type'] not in valid_types:
        raise ValueError(f"Invalid transaction type: {data['type']}")
    
    return True


@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction request"""
    try:
        # Get form data
        data = {
            'step': int(request.form['step']),
            'type': request.form['type'],
            'amount': float(request.form['amount']),
            'nameOrig': request.form['nameOrig'],
            'oldbalanceOrg': float(request.form['oldbalanceOrg']),
            'newbalanceOrig': float(request.form['newbalanceOrig']),
            'nameDest': request.form['nameDest'],
            'oldbalanceDest': float(request.form['oldbalanceDest']),
            'newbalanceDest': float(request.form['newbalanceDest']),
            'isFlaggedFraud': int(request.form['isFlaggedFraud'])
        }
        
        # Validate data
        validate_transaction_data(data)
        
        # Preprocess
        X = preprocess_input(data)
        
        # Predict
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0][1]
        
        # Determine risk level
        if probability < 0.3:
            risk_level = 'LOW'
            risk_color = 'success'
        elif probability < 0.7:
            risk_level = 'MEDIUM'
            risk_color = 'warning'
        else:
            risk_level = 'HIGH'
            risk_color = 'danger'
        
        # Recommendation
        if prediction == 1:
            if probability > 0.9:
                recommendation = 'BLOCK TRANSACTION - High confidence fraud detected'
            else:
                recommendation = 'REVIEW REQUIRED - Suspicious activity detected'
        else:
            recommendation = 'APPROVE - Transaction appears legitimate'
        
        result = {
            'success': True,
            'is_fraud': bool(prediction),
            'probability': float(probability),
            'risk_level': risk_level,
            'risk_color': risk_color,
            'recommendation': recommendation
        }
        
        return jsonify(result)
    
    except ValueError as e:
        return jsonify({'success': False, 'error': str(e)}), 400
    except Exception as e:
        return jsonify({'success': False, 'error': 'Internal server error'}), 500


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    try:
        data = request.get_json()
        X = preprocess_input(data)
        
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0][1]
        
        return jsonify({
            'is_fraud': bool(prediction),
            'fraud_probability': float(probability),
            'risk_level': 'HIGH' if probability > 0.7 else 'MEDIUM' if probability > 0.3 else 'LOW'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    load_models()
    app.run(debug=True, host='127.0.0.1', port=8080)
