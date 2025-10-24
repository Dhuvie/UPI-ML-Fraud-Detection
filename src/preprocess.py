"""
Data preprocessing for fraud detection
Handles missing values, encoding, scaling, and class imbalance
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.under_sampling import RandomUnderSampler
import joblib
import os


def load_data():
    """Load raw transaction data"""
    df = pd.read_csv('data/raw/upi_fraud_data.csv')
    print(f"Loaded {len(df)} transactions")
    print(f"Fraud cases: {df['isFraud'].sum()} ({df['isFraud'].mean()*100:.2f}%)")
    return df


def clean_data(df):
    """Handle missing values and duplicates"""
    # Remove duplicates
    initial_len = len(df)
    df = df.drop_duplicates()
    print(f"Removed {initial_len - len(df)} duplicates")
    
    # Handle missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
    
    return df


def engineer_features(df):
    """Create additional features from transaction data"""
    # Time-based features
    if 'step' in df.columns:
        df['hour'] = df['step'] % 24
        df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
    
    # Amount features
    if 'amount' in df.columns:
        df['amount_log'] = np.log1p(df['amount'])
        df['is_round_amount'] = (df['amount'] % 1000 == 0).astype(int)
    
    # Balance features
    if 'oldbalanceOrg' in df.columns and 'newbalanceOrig' in df.columns:
        df['balance_change_orig'] = df['newbalanceOrig'] - df['oldbalanceOrg']
    
    if 'oldbalanceDest' in df.columns and 'newbalanceDest' in df.columns:
        df['balance_change_dest'] = df['newbalanceDest'] - df['oldbalanceDest']
    
    return df


def encode_features(df):
    """Encode categorical variables"""
    label_encoders = {}
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    
    return df, label_encoders


def balance_dataset(X, y):
    """
    Balance dataset using undersampling
    Keeps all fraud cases and samples legitimate cases to 3:1 ratio
    """
    print("\nBalancing dataset...")
    print(f"Original: {len(y)} samples, {sum(y)} frauds ({sum(y)/len(y)*100:.2f}%)")
    
    undersampler = RandomUnderSampler(
        sampling_strategy=0.33,  # 3 legitimate for every 1 fraud (more balanced)
        random_state=42
    )
    
    X_balanced, y_balanced = undersampler.fit_resample(X, y)
    
    print(f"Balanced: {len(y_balanced)} samples, {sum(y_balanced)} frauds ({sum(y_balanced)/len(y_balanced)*100:.2f}%)")
    
    return X_balanced, y_balanced


def preprocess_data():
    """Main preprocessing pipeline"""
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Load and clean
    df = load_data()
    df = clean_data(df)
    
    # Separate features and target
    target_col = 'isFraud'
    X = df.drop(columns=[target_col])
    y = df[target_col].values
    
    # Engineer features
    X = engineer_features(X)
    
    # Encode categorical features
    X, label_encoders = encode_features(X)
    
    # Split data BEFORE balancing (keep test set imbalanced for realistic evaluation)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Balance only training data
    X_train, y_train = balance_dataset(X_train, y_train)
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Save processed data
    pd.DataFrame(X_train).to_csv('data/processed/X_train.csv', index=False)
    pd.DataFrame(X_test).to_csv('data/processed/X_test.csv', index=False)
    pd.DataFrame(y_train, columns=['target']).to_csv('data/processed/y_train.csv', index=False)
    pd.DataFrame(y_test, columns=['target']).to_csv('data/processed/y_test.csv', index=False)
    
    # Save preprocessing objects
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(label_encoders, 'models/label_encoders.pkl')
    
    print(f"\nPreprocessing complete")
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")


if __name__ == "__main__":
    preprocess_data()
