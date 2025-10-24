"""
Cross-validation for more reliable performance estimates
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from src.train import get_models
import time


def cross_validate_models(n_folds=5):
    """
    Perform k-fold cross-validation on all models
    
    Args:
        n_folds: Number of folds for cross-validation
    
    Returns:
        DataFrame with cross-validation results
    """
    # Load training data
    X_train = pd.read_csv('data/processed/X_train.csv').values
    y_train = pd.read_csv('data/processed/y_train.csv')['target'].values
    
    print(f"Performing {n_folds}-fold cross-validation...")
    print(f"Training samples: {len(y_train)}\n")
    
    # Initialize models
    models = get_models()
    
    # Setup cross-validation
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    results = []
    
    for model_name, model in models.items():
        print(f"Cross-validating {model_name}...")
        
        fold_scores = {
            'precision': [],
            'recall': [],
            'f1': [],
            'roc_auc': []
        }
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
            X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
            y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
            
            # Train model
            model.fit(X_fold_train, y_fold_train)
            
            # Predict
            y_pred = model.predict(X_fold_val)
            y_proba = model.predict_proba(X_fold_val)[:, 1]
            
            # Calculate metrics
            fold_scores['precision'].append(precision_score(y_fold_val, y_pred, zero_division=0))
            fold_scores['recall'].append(recall_score(y_fold_val, y_pred, zero_division=0))
            fold_scores['f1'].append(f1_score(y_fold_val, y_pred, zero_division=0))
            fold_scores['roc_auc'].append(roc_auc_score(y_fold_val, y_proba))
        
        # Calculate mean and std
        result = {
            'Model': model_name,
            'Precision_mean': np.mean(fold_scores['precision']),
            'Precision_std': np.std(fold_scores['precision']),
            'Recall_mean': np.mean(fold_scores['recall']),
            'Recall_std': np.std(fold_scores['recall']),
            'F1_mean': np.mean(fold_scores['f1']),
            'F1_std': np.std(fold_scores['f1']),
            'ROC_AUC_mean': np.mean(fold_scores['roc_auc']),
            'ROC_AUC_std': np.std(fold_scores['roc_auc'])
        }
        
        results.append(result)
        
        print(f"  F1-Score: {result['F1_mean']:.3f} ± {result['F1_std']:.3f}")
    
    # Create results dataframe
    df = pd.DataFrame(results)
    df = df.sort_values('F1_mean', ascending=False)
    
    # Save results
    df.to_csv('results/cross_validation.csv', index=False)
    print(f"\nCross-validation results saved to results/cross_validation.csv")
    
    return df


if __name__ == "__main__":
    cross_validate_models(n_folds=5)
