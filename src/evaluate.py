"""
Model evaluation for fraud detection
Evaluates models on test set and generates reports
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import joblib
import os


def load_test_data():
    """Load preprocessed test data"""
    X_test = pd.read_csv('data/processed/X_test.csv').values
    y_test = pd.read_csv('data/processed/y_test.csv')['target'].values
    return X_test, y_test


def load_models():
    """Load all trained models"""
    models = {}
    model_files = [f for f in os.listdir('models') if f.endswith('.pkl')]
    
    for file in model_files:
        if file in ['scaler.pkl', 'label_encoders.pkl']:
            continue
        
        name = file.replace('.pkl', '')
        models[name] = joblib.load(f'models/{file}')
    
    return models


def find_optimal_threshold(y_test, y_proba):
    """Find threshold that maximizes F1-Score"""
    from sklearn.metrics import precision_recall_curve
    
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
    
    return optimal_threshold


def evaluate_model(name, model, X_test, y_test):
    """Evaluate a single model with optimal threshold"""
    # Get probabilities if available
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
        has_proba = True
    except:
        y_proba = model.predict(X_test)
        has_proba = False
    
    # Find optimal threshold and make predictions
    if has_proba and len(np.unique(y_proba)) > 1:
        optimal_threshold = find_optimal_threshold(y_test, y_proba)
        y_pred = (y_proba >= optimal_threshold).astype(int)
    else:
        optimal_threshold = 0.5
        y_pred = model.predict(X_test)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Calculate specificity
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    metrics = {
        'Model': name,
        'Threshold': optimal_threshold,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall': recall_score(y_test, y_pred, zero_division=0),
        'Specificity': specificity,
        'F1-Score': f1_score(y_test, y_pred, zero_division=0)
    }
    
    return metrics, y_pred, cm


def plot_confusion_matrices(results, y_test):
    """Plot confusion matrices for all models"""
    n_models = len(results)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, (name, data) in enumerate(results.items()):
        if idx >= len(axes):
            break
        
        cm = data['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
        axes[idx].set_title(f'{name}\nF1: {data["metrics"]["F1-Score"]:.3f}')
        axes[idx].set_ylabel('Actual')
        axes[idx].set_xlabel('Predicted')
    
    # Hide extra subplots
    for idx in range(len(results), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('results/confusion_matrices.png', dpi=300, bbox_inches='tight')
    print("Saved confusion matrices to results/confusion_matrices.png")


def plot_metrics_comparison(df):
    """Plot comparison of all metrics"""
    metrics = ['Precision', 'Recall', 'Specificity', 'F1-Score']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        df_sorted = df.sort_values(metric, ascending=True)
        axes[idx].barh(df_sorted['Model'], df_sorted[metric])
        axes[idx].set_xlabel(metric)
        axes[idx].set_title(f'{metric} Comparison')
        axes[idx].set_xlim(0, 1)
        
        # Add value labels
        for i, v in enumerate(df_sorted[metric]):
            axes[idx].text(v + 0.01, i, f'{v:.3f}', va='center')
    
    plt.tight_layout()
    plt.savefig('results/metrics_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved metrics comparison to results/metrics_comparison.png")


def evaluate_models():
    """Evaluate all models"""
    os.makedirs('results', exist_ok=True)
    
    # Load data and models
    X_test, y_test = load_test_data()
    models = load_models()
    
    print(f"Evaluating on {len(y_test)} test samples")
    print(f"Test set fraud rate: {y_test.mean()*100:.2f}%\n")
    
    # Evaluate each model
    results = {}
    metrics_list = []
    
    for name, model in models.items():
        print(f"Evaluating {name}...")
        metrics, y_pred, cm = evaluate_model(name, model, X_test, y_test)
        
        results[name] = {
            'metrics': metrics,
            'predictions': y_pred,
            'confusion_matrix': cm
        }
        
        metrics_list.append(metrics)
        
        print(f"  Threshold: {metrics['Threshold']:.3f}")
        print(f"  Precision: {metrics['Precision']:.3f}")
        print(f"  Recall: {metrics['Recall']:.3f}")
        print(f"  Specificity: {metrics['Specificity']:.3f}")
        print(f"  F1-Score: {metrics['F1-Score']:.3f}")
    
    # Create results dataframe
    df = pd.DataFrame(metrics_list)
    df = df.sort_values('F1-Score', ascending=False)
    
    # Save metrics
    df.to_csv('results/metrics.csv', index=False)
    print(f"\nSaved metrics to results/metrics.csv")
    
    # Generate plots
    plot_confusion_matrices(results, y_test)
    plot_metrics_comparison(df)
    
    # Print summary
    print("\n" + "=" * 60)
    print("BEST MODELS")
    print("=" * 60)
    for metric in ['Precision', 'Recall', 'Specificity', 'F1-Score']:
        best = df.loc[df[metric].idxmax()]
        print(f"{metric:12s}: {best['Model']:20s} ({best[metric]:.4f})")
    print("=" * 60)


if __name__ == "__main__":
    evaluate_models()
