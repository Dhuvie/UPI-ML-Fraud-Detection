"""
Prediction module for real-time fraud detection
Load trained model and make predictions on new transactions
"""

import pandas as pd
import numpy as np
import joblib


class FraudDetector:
    """Real-time fraud detection using trained model"""
    
    def __init__(self, model_name='xgboost'):
        """Load trained model and preprocessing objects"""
        self.model = joblib.load(f'models/{model_name}.pkl')
        self.scaler = joblib.load('models/scaler.pkl')
        self.label_encoders = joblib.load('models/label_encoders.pkl')
        self.model_name = model_name
        print(f"Loaded {model_name} model for predictions")
    
    def preprocess_transaction(self, transaction):
        """Preprocess a single transaction"""
        df = pd.DataFrame([transaction])
        
        # Encode categorical features
        for col, encoder in self.label_encoders.items():
            if col in df.columns:
                try:
                    df[col] = encoder.transform(df[col].astype(str))
                except:
                    df[col] = 0  # Unknown category
        
        # Scale features
        X = self.scaler.transform(df.values)
        
        return X
    
    def predict(self, transaction):
        """
        Predict if a transaction is fraudulent
        
        Args:
            transaction: Dictionary with transaction details
        
        Returns:
            Dictionary with prediction and probability
        """
        X = self.preprocess_transaction(transaction)
        
        # Get prediction and probability
        prediction = self.model.predict(X)[0]
        probability = self.model.predict_proba(X)[0][1]
        
        result = {
            'is_fraud': bool(prediction),
            'fraud_probability': float(probability),
            'risk_level': self._get_risk_level(probability),
            'recommendation': self._get_recommendation(prediction, probability)
        }
        
        return result
    
    def _get_risk_level(self, probability):
        """Categorize risk level based on probability"""
        if probability < 0.3:
            return 'LOW'
        elif probability < 0.7:
            return 'MEDIUM'
        else:
            return 'HIGH'
    
    def _get_recommendation(self, prediction, probability):
        """Provide recommendation based on prediction"""
        if prediction == 1:
            if probability > 0.9:
                return 'BLOCK - High confidence fraud'
            else:
                return 'REVIEW - Suspicious transaction'
        else:
            return 'APPROVE - Legitimate transaction'
    
    def predict_batch(self, transactions):
        """Predict multiple transactions"""
        results = []
        for transaction in transactions:
            result = self.predict(transaction)
            results.append(result)
        return results


def example_usage():
    """Example of how to use the fraud detector"""
    # Initialize detector
    detector = FraudDetector(model_name='xgboost')
    
    # Example transaction
    transaction = {
        'step': 1,
        'type': 'PAYMENT',
        'amount': 9000.60,
        'nameOrig': 'C1231006815',
        'oldbalanceOrg': 170136.0,
        'newbalanceOrig': 161136.0,
        'nameDest': 'M1979787155',
        'oldbalanceDest': 0.0,
        'newbalanceDest': 0.0,
        'isFlaggedFraud': 0
    }
    
    # Make prediction
    result = detector.predict(transaction)
    
    print("\nTransaction Analysis:")
    print(f"Fraud: {result['is_fraud']}")
    print(f"Probability: {result['fraud_probability']:.2%}")
    print(f"Risk Level: {result['risk_level']}")
    print(f"Recommendation: {result['recommendation']}")
    
    return result


if __name__ == "__main__":
    example_usage()
