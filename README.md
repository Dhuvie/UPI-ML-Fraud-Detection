# UPI Fraud Detection System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0%2B-green)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Machine learning system for detecting fraudulent UPI transactions with **87% F1-Score** using XGBoost and advanced feature engineering.

## Performance

| Model | Precision | Recall | F1-Score | Specificity |
|-------|-----------|--------|----------|-------------|
| **XGBoost** | 90.5% | 83.8% | **87.0%** | 99.98% |
| Random Forest | 94.6% | 79.5% | 86.4% | 99.99% |
| LightGBM | 87.8% | 79.9% | 83.6% | 99.97% |
| Gradient Boosting | 80.6% | 82.7% | 81.6% | 99.95% |

### Understanding the Models

#### XGBoost (Extreme Gradient Boosting) - *Best Overall*

**What it does:**
XGBoost is like a team of detectives where each new detective learns from the mistakes of previous ones. It builds multiple "decision trees" (think of them as flowcharts) that work together to identify fraud.

**How it works:**
1. First tree makes predictions (some right, some wrong)
2. Second tree focuses on fixing the mistakes of the first
3. Third tree fixes mistakes of the second, and so on
4. Final prediction combines all trees' opinions

**Real-world example:**
- Transaction 1: Amount = ₹50,000, Time = 2 AM, Balance drops to ₹0
- Tree 1 says: "Suspicious amount" → 60% fraud
- Tree 2 adds: "Plus it's night time" → 75% fraud
- Tree 3 adds: "And balance went to zero" → 92% fraud
- **Final decision: FRAUD** 

**Performance:**
- 90.5% Precision = Out of 100 flagged frauds, 90 are real
- 83.8% Recall = Catches 84 out of 100 actual frauds
- **Best for production use**

---

#### Random Forest - *Most Precise*

**What it does:**
Imagine asking 200 fraud experts to independently review a transaction. Each expert looks at different aspects (amount, time, location, history). The majority vote wins.

**How it works:**
1. Creates 200 independent decision trees
2. Each tree sees a random sample of data
3. Each tree votes: Fraud or Legitimate
4. Final decision = majority vote

**Real-world example:**
- Transaction: ₹10,000 payment at 3 PM
- 180 trees say: "Legitimate" (normal amount, normal time)
- 20 trees say: "Fraud" (suspicious pattern)
- **Final decision: LEGITIMATE** (180 > 20) 

**Performance:**
- 94.6% Precision = Only 5 false alarms per 100 flags
- **Best when you can't afford false alarms** (blocking legitimate customers is costly)

**Why it's precise:**
- Multiple independent opinions reduce errors
- Doesn't overreact to single suspicious features
- Conservative approach = fewer mistakes

---

#### LightGBM (Light Gradient Boosting Machine) - *Fastest*

**What it does:**
LightGBM is like XGBoost's faster cousin. It uses smart shortcuts to analyze millions of transactions quickly without sacrificing much accuracy.

**How it works:**
1. Instead of checking every transaction, it groups similar ones
2. Builds trees "leaf-wise" (grows the most useful branches first)
3. Uses histogram-based learning (buckets data for speed)

**Real-world example:**
- Dataset: 6 million transactions
- XGBoost: Takes 10 minutes to train
- LightGBM: Takes 3 minutes to train
- **Same accuracy, 3x faster** 

**Performance:**
- 87.8% Precision, 79.9% Recall
- **Best for real-time systems** where speed matters
- Can handle streaming data (thousands of transactions per second)

**When to use:**
- Large datasets (millions of rows)
- Real-time fraud detection
- Limited computing resources

---

#### Gradient Boosting - *Reliable Baseline*

**What it does:**
Builds a team of "weak learners" (simple models) that together become a "strong learner". Each new model focuses on the hardest cases the previous models got wrong.

**How it works:**
1. Start with simple predictions (average fraud rate)
2. Find transactions where prediction was wrong
3. Build new model to fix those specific errors
4. Repeat 100-200 times
5. Combine all models

**Real-world example:**
- Model 1: Flags high amounts → Catches 50% of frauds
- Model 2: Adds time patterns → Catches 65% of frauds
- Model 3: Adds balance changes → Catches 75% of frauds
- Model 100: **Catches 82.7% of frauds** 

**Performance:**
- 80.6% Precision, 82.7% Recall
- **Most balanced recall** (catches the most frauds)
- Stable and consistent across different datasets

**Why we use it:**
- Benchmark to compare other models
- Reliable fallback option
- Good all-around performance

---

#### Logistic Regression - *Simple & Interpretable*

**What it does:**
Uses a mathematical formula to calculate fraud probability. Assigns a "weight" to each feature based on how important it is for detecting fraud.

**How it works:**
```
Fraud Score = (Amount × 0.3) + (Night × 0.5) + (Balance_Drop × 0.8) - 2.0

If Score > 0 → Fraud
If Score < 0 → Legitimate
```

**Real-world example:**
Transaction: ₹100,000 at 3 AM, balance drops from ₹100,000 to ₹0

```
Score = (100000 × 0.00003) + (1 × 0.5) + (100000 × 0.00008) - 2.0
      = 3.0 + 0.5 + 8.0 - 2.0
      = 9.5 (> 0)
→ FRAUD 
```

**Performance:**
- 81.1% Precision, 49.8% Recall
- **Best for explaining decisions** to regulators, auditors, customers

**Why it's interpretable:**
- Can see exact contribution of each feature
- "Your transaction was flagged because: high amount (30%), night time (50%), balance drop (80%)"
- Easy to audit and explain
- Meets regulatory requirements

**When to use:**
- Need to explain why transaction was flagged
- Regulatory compliance
- Customer disputes
- Building trust with stakeholders

### Metrics Explained

**Classification Metrics:**

- **Precision**: Of all transactions flagged as fraud, how many were actually fraud?
  - Example: 90.5% = Out of 100 flagged transactions, 90 are real fraud, only 10 false alarms
  - *Why it matters*: High precision means fewer legitimate customers get blocked
  - *Business impact*: Reduces customer complaints and support costs

- **Recall**: Of all actual fraud cases, how many did we catch?
  - Example: 83.8% = We catch 84 out of 100 real fraud cases
  - *Why it matters*: High recall means we don't miss many frauds
  - *Business impact*: Prevents financial losses from undetected fraud

- **F1-Score**: Harmonic mean of precision and recall (balanced measure)
  - Example: 87% = Excellent balance between catching frauds and avoiding false alarms
  - *Why it matters*: Single metric to compare models
  - *Business impact*: Optimal trade-off between security and user experience

- **Specificity**: Of all legitimate transactions, how many did we correctly identify?
  - Example: 99.98% = Out of 10,000 legitimate transactions, we correctly identify 9,998
  - *Why it matters*: Shows we rarely flag legitimate transactions as fraud
  - *Business impact*: Maintains customer trust and satisfaction

**Real-World Impact:**
- **High Precision (90.5%)**: Only 1 in 10 fraud alerts is a false alarm → Less manual review needed
- **High Recall (83.8%)**: We catch 5 out of 6 fraud attempts → Saves millions in fraud losses
- **High Specificity (99.98%)**: Only 2 in 10,000 legitimate users get blocked → Happy customers
- **High F1-Score (87%)**: Best overall balance for production use → Ready to deploy

## Features

- **Web Interface** - User-friendly fraud detection dashboard
- **High Accuracy** - 87% F1-Score with 90.5% precision
- **Real-time Predictions** - Instant fraud detection
- **Feature Engineering** - Automated feature creation
- **Imbalanced Data Handling** - Undersampling with optimal thresholds
- **Multiple Models** - 5 ML algorithms trained and compared

## Quick Start

### 1. Installation

```bash
git clone https://github.com/Dhuvie/UPI-ML-Fraud-Detection.git
cd UPI-ML-Fraud-Detection
pip install -r requirements.txt

# For Jupyter notebooks
pip install jupyter
```

### 2. Get Dataset

Download from Kaggle:
- [PaySim Mobile Money Simulator](https://www.kaggle.com/datasets/ealaxi/paysim1)
- [Online Payment Fraud Detection](https://www.kaggle.com/datasets/jainilcoder/online-payment-fraud-detection)

Place CSV file in: `data/raw/upi_fraud_data.csv`

### 3. Train Models

```bash
python main.py
```

This will:
- Preprocess data and engineer features
- Train 5 ML models
- Evaluate and save results

### 4. Launch Web Interface

```bash
python app.py
```

Open browser: `http://localhost:8080`

## Usage

### Web Interface

The web app provides:
- Transaction input form
- Real-time fraud detection
- Fraud probability (0-100%)
- Risk level (LOW/MEDIUM/HIGH)
- Actionable recommendations

**Example Transaction:**
```
Type: PAYMENT
Amount: ₹10,000
Sender: C1234567890
Sender Old Balance: ₹50,000
Sender New Balance: ₹40,000
Receiver: M9876543210
Receiver Old Balance: ₹0
Receiver New Balance: ₹10,000
```

### Interactive Learning (Jupyter Notebooks)

Explore the project interactively:

```bash
jupyter notebook
```

**Available Notebooks:**
1. `01_data_exploration.ipynb` - Explore dataset and fraud patterns
2. `02_feature_engineering.ipynb` - Learn feature creation techniques
3. `03_model_training.ipynb` - Train and compare models

### Python API

```python
from src.predict import FraudDetector

detector = FraudDetector('xgboost')
result = detector.predict(transaction)

print(f"Fraud: {result['is_fraud']}")
print(f"Probability: {result['fraud_probability']:.2%}")
print(f"Risk: {result['risk_level']}")
```

### Individual Components

```bash
# Preprocess data
python src/preprocess.py

# Train models
python src/train.py

# Evaluate models
python src/evaluate.py

# Explain predictions
python src/explain.py

# Cross-validation
python src/validate.py
```

## Project Structure

```
├── app.py                    # Flask web application
├── main.py                   # Complete ML pipeline
├── requirements.txt          # Dependencies
├── data/
│   ├── raw/                 # Original dataset
│   └── processed/           # Preprocessed data
├── models/                   # Trained models
├── results/                  # Metrics and plots
├── notebooks/                # Jupyter notebooks for learning
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_training.ipynb
├── src/
│   ├── preprocess.py        # Data preprocessing + feature engineering
│   ├── train.py             # Model training
│   ├── evaluate.py          # Model evaluation
│   ├── predict.py           # Real-time predictions
│   ├── explain.py           # Model explainability
│   └── validate.py          # Cross-validation
└── templates/
    └── index.html           # Web interface
```

## Technical Details

### Feature Engineering

Automatically creates:
- **Time features**: Hour of day, night indicator
- **Amount features**: Log transformation, round amount detection
- **Balance features**: Balance changes for sender/receiver

### Data Balancing

- **Training set**: Undersampled to 3:1 ratio (25% fraud)
- **Test set**: Kept imbalanced (0.13% fraud) for realistic evaluation
- **Why**: Models learn fraud patterns better with balanced training data

### Threshold Optimization

- Default 0.5 threshold → Optimized threshold (0.94-0.99)
- Maximizes F1-Score for each model
- **Impact**: Precision improved from 15% to 90.5%

### Model Training

Models trained with:
- Class weights for imbalance handling
- Regularization (subsample, colsample)
- Optimal hyperparameters
- 5-fold cross-validation

## Results

After running `python main.py`, check:
- `results/metrics.csv` - Performance comparison
- `results/confusion_matrices.png` - Visual confusion matrices
- `results/metrics_comparison.png` - Metrics bar charts
- `results/feature_importance_xgboost.png` - Feature importance

## API Endpoint

REST API for programmatic access:

```bash
curl -X POST http://localhost:8080/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "step": 1,
    "type": "PAYMENT",
    "amount": 9000.60,
    "nameOrig": "C1231006815",
    "oldbalanceOrg": 170136.0,
    "newbalanceOrig": 161136.0,
    "nameDest": "M1979787155",
    "oldbalanceDest": 0.0,
    "newbalanceDest": 0.0,
    "isFlaggedFraud": 0
  }'
```

Response:
```json
{
  "is_fraud": false,
  "fraud_probability": 0.023,
  "risk_level": "LOW"
}
```

## Requirements

- Python 3.8+
- numpy
- pandas
- scikit-learn
- xgboost
- lightgbm
- matplotlib
- seaborn
- imbalanced-learn
- joblib
- flask

## Security

The application includes:
- Input validation (negative amounts rejected)
- Transaction type whitelisting
- Error handling
- Request size limits (16MB)

## Troubleshooting

**Models not found:**
```
Error: Models not found. Run 'python main.py' first.
```
Solution: Train models with `python main.py`

**Port already in use:**
```
Address already in use
```
Solution: Change port in `app.py` line 199 to `port=3000` or `port=9000`

**Import errors:**
```
ModuleNotFoundError: No module named 'flask'
```
Solution: `pip install -r requirements.txt`

## Contributing

Contributions are welcome! Here's how you can help:

### How to Contribute

1. **Fork the repository**
   ```bash
   git clone https://github.com/Dhuvie/UPI-ML-Fraud-Detection.git
   cd UPI-ML-Fraud-Detection
   ```

2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**
   - Follow existing code style
   - Add comments and docstrings
   - Test your changes

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "Add: your feature description"
   ```

5. **Push and create Pull Request**
   ```bash
   git push origin feature/your-feature-name
   ```

### Adding New Features

#### 1. Adding a New ML Model

Create a new model in `src/train.py`:

```python
from sklearn.ensemble import AdaBoostClassifier

# Add to models dictionary
models = {
    'adaboost': AdaBoostClassifier(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42
    )
}
```

#### 2. Adding New Features

Add feature engineering in `src/preprocess.py`:

```python
def engineer_features(df):
    # Your new feature
    df['transaction_velocity'] = df.groupby('nameOrig')['step'].diff()
    df['amount_to_balance_ratio'] = df['amount'] / (df['oldbalanceOrg'] + 1)
    return df
```

#### 3. Adding API Endpoints

Add new routes in `app.py`:

```python
@app.route('/api/batch_predict', methods=['POST'])
def batch_predict():
    """Predict multiple transactions"""
    transactions = request.get_json()
    results = [predict_single(t) for t in transactions]
    return jsonify(results)
```

#### 4. Adding Visualizations

Create new plots in `src/explain.py`:

```python
def plot_fraud_timeline(df):
    """Plot fraud cases over time"""
    fraud_by_time = df[df['isFraud']==1].groupby('step').size()
    plt.plot(fraud_by_time)
    plt.title('Fraud Cases Over Time')
    plt.savefig('results/fraud_timeline.png')
```

### Code Guidelines

- **Style**: Follow PEP 8
- **Docstrings**: Add to all functions
- **Comments**: Explain complex logic
- **Testing**: Test before submitting
- **Commits**: Use clear, descriptive messages

### Commit Message Format

```
Type: Brief description

- Detailed point 1
- Detailed point 2
```

**Types:**
- `Add:` New feature
- `Fix:` Bug fix
- `Update:` Modify existing feature
- `Remove:` Delete code/feature
- `Docs:` Documentation changes

### Examples

**Good commits:**
```
Add: Transaction velocity feature

- Calculate time between transactions
- Add to feature engineering pipeline
- Improves F1-Score by 2%
```

**Bad commits:**
```
update stuff
fixed bug
changes
```

### What to Contribute

**High Priority:**
- Additional ML models (Neural Networks, SVM)
- Real-time streaming predictions
- Model deployment scripts
- API authentication
- Rate limiting
- Batch prediction endpoint

**Medium Priority:**
- Additional visualizations
- Performance optimizations
- More feature engineering
- Cross-validation improvements
- Hyperparameter tuning

**Documentation:**
- Tutorial notebooks
- API documentation
- Deployment guides
- Use case examples

### Testing Your Changes

Before submitting:

```bash
# Test ML pipeline
python main.py

# Test web app
python app.py

# Test notebooks
jupyter notebook
```

### Questions?

Open an issue for:
- Bug reports
- Feature requests
- Questions
- Discussions

## License

MIT License - see [LICENSE](LICENSE) file

## Author

**Dhuvie** - [GitHub](https://github.com/Dhuvie)

## Acknowledgments

- Kaggle for datasets
- scikit-learn, XGBoost, LightGBM communities
- Flask framework
