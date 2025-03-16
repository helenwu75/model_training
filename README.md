# Model Training Repository for Polymarket Election Analysis

This repository contains code for training and evaluating machine learning models to predict the accuracy of Polymarket election markets. It uses processed data from the Polymarket subgraph analysis project to build models that predict Brier scores, log loss, and binary prediction correctness.

## Repository Overview

This repository focuses solely on the model training and evaluation aspects of the Polymarket election analysis project. It uses preprocessed data to build predictive models for market accuracy.

## Directory Structure

```
model_training/
├── modified_analysis/       # Processed data ready for modeling
├── model_results/           # Model outputs and visualizations
├── data_cleaning.py         # Preprocessing script that creates modified_analysis
├── tests/                   # Test scripts and notebooks
│   ├── binary_model_results/# Binary classification model tests
│   ├── feature_analysis.py  # Feature importance analysis tests
│   └── results.ipynb        # Results visualization notebook
└── data_analytics/          # Additional analysis scripts
    ├── prediction_error_analysis.py  # Analysis of prediction errors
    ├── descriptive_stats.py          # Descriptive statistics generation
    ├── insights.py                   # Actionable insights development
    └── feature_relationship_analysis.py  # Feature correlation analysis
```

## Data Structure in `modified_analysis/`

The `modified_analysis` directory contains preprocessed data files ready for model training:

| File                                   | Description                                              | Format      |
| -------------------------------------- | -------------------------------------------------------- | ----------- |
| `X_train_preprocessed.npy`             | Training features (processed and standardized)           | NumPy array |
| `X_test_preprocessed.npy`              | Test features (processed and standardized)               | NumPy array |
| `y_train_brier_score.npy`              | Training target - Brier score                            | NumPy array |
| `y_test_brier_score.npy`               | Test target - Brier score                                | NumPy array |
| `y_train_log_loss.npy`                 | Training target - Log loss                               | NumPy array |
| `y_test_log_loss.npy`                  | Test target - Log loss                                   | NumPy array |
| `y_train_prediction_correct.npy`       | Training target - Binary prediction correctness          | NumPy array |
| `y_test_prediction_correct.npy`        | Test target - Binary prediction correctness              | NumPy array |
| `X_train_smote.npy`                    | SMOTE-balanced training features for classification      | NumPy array |
| `y_train_prediction_correct_smote.npy` | SMOTE-balanced training target for classification        | NumPy array |
| `X_train_original.csv`                 | Original (pre-standardized) training features            | CSV         |
| `X_test_original.csv`                  | Original (pre-standardized) test features                | CSV         |
| `transformed_feature_names.csv`        | Names of features after one-hot encoding                 | CSV         |
| `original_feature_names.csv`           | Original feature names before preprocessing              | CSV         |
| `train_identifiers.csv`                | Identifiers for training samples (market IDs, questions) | CSV         |
| `test_identifiers.csv`                 | Identifiers for test samples (market IDs, questions)     | CSV         |
| `feature_preprocessor.joblib`          | Sklearn preprocessing pipeline for new data              | Joblib      |
| `complete_election_data.csv`           | Complete dataset with all columns                        | CSV         |
| `cleaned_election_data.csv`            | Cleaned dataset with modeling features and targets       | CSV         |

## Output Structure in `model_results/`

The `model_results` directory contains the outputs from model training:

| File Pattern                        | Description                                           | Format |
| ----------------------------------- | ----------------------------------------------------- | ------ |
| `model_comparison.csv`              | Comparison of all models' performance metrics         | CSV    |
| `model_comparison_roc_curves.png`   | ROC curves for all classification models              | PNG    |
| `rf_feature_importance.csv`         | Random Forest feature importance rankings             | CSV    |
| `rf_feature_importance.png`         | Visualization of Random Forest feature importance     | PNG    |
| `l1_logistic_coefficients.csv`      | Logistic Regression coefficient values                | CSV    |
| `l1_logistic_coefficients.png`      | Visualization of logistic regression coefficients     | PNG    |
| `gb_feature_importance.csv`         | Gradient Boosting feature importance rankings         | CSV    |
| `gb_feature_importance.png`         | Visualization of Gradient Boosting feature importance | PNG    |
| `pca_explained_variance.png`        | Explained variance ratio by PCA components            | PNG    |
| `pca_top_loadings.csv`              | Top feature loadings for each PCA component           | CSV    |
| `shap_summary_*.png`                | SHAP summary plots for best model                     | PNG    |
| `shap_importance_*.csv`             | Feature importance rankings from SHAP analysis        | CSV    |
| `feature_importance_comparison.csv` | Comparison of importance across methods               | CSV    |
| `*_model.joblib`                    | Serialized model files                                | Joblib |

## Key Files

### `data_cleaning.py`

This script prepares the data for modeling:

- Removes features that would cause data leakage
- Imputes missing values
- Groups rare categorical values (election types, countries)
- Splits data into train and test sets
- Applies feature standardization
- Uses SMOTE for class balancing
- Saves processed data to `modified_analysis/`

### `tests/binary_model_results/binary_model_training.py`

This script trains and evaluates classification models:

- Loads preprocessed data from `modified_analysis/`
- Trains multiple model types (Random Forest, Gradient Boosting, Logistic Regression, PCA-based models)
- Evaluates models using appropriate metrics
- Generates feature importance rankings
- Creates visualizations
- Performs SHAP analysis on the best model
- Saves all results to `model_results/`

### `tests/feature_analysis.py`

This script performs comprehensive feature importance analysis:

- Analyzes feature importance using multiple methods
- Calculates permutation importance
- Performs SHAP analysis
- Creates consensus feature rankings
- Generates visualizations of feature relationships

## Target Variables

The models are trained to predict:

1. **Brier Score** (Primary) - Measures squared error of probability predictions (lower is better)
2. **Log Loss** (Secondary) - Measures negative log-likelihood of outcomes (lower is better)
3. **Prediction Correctness** (Binary) - Whether the market prediction was correct (1) or incorrect (0)

## Feature Categories

The models use features from four main categories:

1. **Market Activity**: volume, trading frequency, trading continuity, etc.
2. **Price Dynamics**: volatility, range, momentum, fluctuations
3. **Trader Behavior**: unique traders count, trader concentration, two-way traders ratio
4. **Market Context**: event type, country, comment metrics (properly treated as categorical/continuous)

## Usage

To run the pipeline:

1. **Preprocess Data**:

   ```bash
   python data_cleaning.py
   ```

2. **Train Classification Models**:

   ```bash
   python tests/binary_model_results/binary_model_training.py
   ```

3. **Analyze Feature Importance**:

   ```bash
   python tests/feature_analysis.py
   ```

4. **Generate Descriptive Statistics**:

   ```bash
   python data_analytics/descriptive_stats.py
   ```

5. **Analyze Prediction Errors**:

   ```bash
   python data_analytics/prediction_error_analysis.py
   ```

6. **Develop Actionable Insights**:
   ```bash
   python data_analytics/insights.py
   ```

## Model Performance Summary

Based on the latest results, the models show:

- High classification accuracy (~95.9% for Gradient Boosting)
- Strong ROC AUC scores (~0.960 for Gradient Boosting)
- Key predictive features include price volatility, price range, trading continuity, and new trader influx
- Class imbalance is well-handled through SMOTE preprocessing (8 incorrect vs. 139 correct predictions in test set)

## Installation

```bash
pip install -r requirements.txt
```

## Author

Helen Wu
