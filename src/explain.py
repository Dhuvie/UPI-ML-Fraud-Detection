"""
Model explainability and feature importance
Helps understand what drives fraud predictions
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os


def plot_feature_importance(model_name='xgboost'):
    """Plot feature importance for tree-based models"""
    model = joblib.load(f'models/{model_name}.pkl')
    
    # Get feature importance
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        
        # Load feature names from processed data
        X_train = pd.read_csv('data/processed/X_train.csv')
        features = X_train.columns
        
        # Create dataframe
        importance_df = pd.DataFrame({
            'feature': features,
            'importance': importance
        }).sort_values('importance', ascending=False).head(20)
        
        # Plot
        plt.figure(figsize=(10, 8))
        plt.barh(importance_df['feature'], importance_df['importance'])
        plt.xlabel('Importance')
        plt.title(f'Top 20 Features - {model_name}')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(f'results/feature_importance_{model_name}.png', dpi=300)
        print(f"Saved feature importance to results/feature_importance_{model_name}.png")
        
        return importance_df
    else:
        print(f"{model_name} does not support feature importance")
        return None


def analyze_predictions(model_name='xgboost', n_samples=100):
    """Analyze model predictions on test set"""
    # Load model and data
    model = joblib.load(f'models/{model_name}.pkl')
    X_test = pd.read_csv('data/processed/X_test.csv')
    y_test = pd.read_csv('data/processed/y_test.csv')['target'].values
    
    # Get predictions and probabilities
    y_pred = model.predict(X_test.values)
    y_proba = model.predict_proba(X_test.values)[:, 1]
    
    # Create analysis dataframe
    analysis = pd.DataFrame({
        'actual': y_test,
        'predicted': y_pred,
        'probability': y_proba,
        'correct': (y_test == y_pred).astype(int)
    })
    
    # Analyze false positives and false negatives
    false_positives = analysis[(analysis['actual'] == 0) & (analysis['predicted'] == 1)]
    false_negatives = analysis[(analysis['actual'] == 1) & (analysis['predicted'] == 0)]
    
    print(f"\nPrediction Analysis for {model_name}:")
    print(f"False Positives: {len(false_positives)} ({len(false_positives)/len(analysis)*100:.2f}%)")
    print(f"False Negatives: {len(false_negatives)} ({len(false_negatives)/len(analysis)*100:.2f}%)")
    
    # Save analysis
    analysis.to_csv(f'results/prediction_analysis_{model_name}.csv', index=False)
    print(f"Saved analysis to results/prediction_analysis_{model_name}.csv")
    
    return analysis


if __name__ == "__main__":
    os.makedirs('results', exist_ok=True)
    
    # Analyze best model (XGBoost)
    print("Analyzing XGBoost model...")
    plot_feature_importance('xgboost')
    analyze_predictions('xgboost')
