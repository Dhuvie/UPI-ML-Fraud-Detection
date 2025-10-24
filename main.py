"""
Main pipeline for UPI fraud detection
Runs preprocessing, training, and evaluation
"""

import os
import sys
from src.preprocess import preprocess_data
from src.train import train_models
from src.evaluate import evaluate_models


def main():
    print("UPI Fraud Detection Pipeline")
    print("-" * 50)
    
    # Check if raw data exists
    if not os.path.exists('data/raw/upi_fraud_data.csv'):
        print("Error: Dataset not found at data/raw/upi_fraud_data.csv")
        print("Please download a dataset from Kaggle and place it in data/raw/")
        return
    
    # Step 1: Preprocess data
    print("\n[1/3] Preprocessing data...")
    try:
        preprocess_data()
    except Exception as e:
        print(f"Preprocessing failed: {e}")
        return
    
    # Step 2: Train models
    print("\n[2/3] Training models...")
    try:
        train_models()
    except Exception as e:
        print(f"Training failed: {e}")
        return
    
    # Step 3: Evaluate models
    print("\n[3/3] Evaluating models...")
    try:
        evaluate_models()
    except Exception as e:
        print(f"Evaluation failed: {e}")
        return
    
    print("\n" + "=" * 50)
    print("Pipeline completed successfully!")
    print("Check results/ directory for metrics and plots")
    print("=" * 50)


if __name__ == "__main__":
    main()
