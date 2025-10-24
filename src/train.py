"""
Model training for fraud detection
Trains multiple classifiers with class weights
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import joblib
import os
import time


def load_training_data():
    """Load preprocessed training data"""
    X_train = pd.read_csv('data/processed/X_train.csv').values
    y_train = pd.read_csv('data/processed/y_train.csv')['target'].values
    return X_train, y_train


def get_models():
    """Initialize models with optimized parameters"""
    models = {
        'random_forest': RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=4,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ),
        'xgboost': XGBClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.05,
            scale_pos_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss'
        ),
        'lightgbm': LGBMClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.05,
            num_leaves=50,
            class_weight='balanced',
            random_state=42,
            verbose=-1
        ),
        'gradient_boosting': GradientBoostingClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42
        ),
        'logistic_regression': LogisticRegression(
            max_iter=1000,
            C=0.1,
            class_weight='balanced',
            random_state=42
        )
    }
    return models


def train_models():
    """Train all models"""
    os.makedirs('models', exist_ok=True)
    
    # Load data
    X_train, y_train = load_training_data()
    print(f"Training on {len(y_train)} samples")
    
    # Get models
    models = get_models()
    
    # Train each model
    for name, model in models.items():
        print(f"\nTraining {name}...")
        start = time.time()
        
        model.fit(X_train, y_train)
        
        elapsed = time.time() - start
        print(f"Completed in {elapsed:.2f}s")
        
        # Save model
        joblib.dump(model, f'models/{name}.pkl')
    
    print(f"\nAll models trained and saved to models/")


if __name__ == "__main__":
    train_models()
