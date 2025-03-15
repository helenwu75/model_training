import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Create output directory
output_dir = "prediction_error_analysis"
os.makedirs(output_dir, exist_ok=True)

# Load your data
df = pd.read_csv("modified_analysis/cleaned_election_data.csv")
X_test = np.load("modified_analysis/X_test_preprocessed.npy")
y_test = np.load("modified_analysis/y_test_brier_score.npy")

# Load identifiers for test data
test_ids = pd.read_csv("modified_analysis/test_identifiers.csv")

# Load your best model (Random Forest based on your results)
best_model = joblib.load("feature_importance_results/random_forest_model.joblib")

# Get predictions
y_pred = best_model.predict(X_test)

# Calculate errors
errors = y_test - y_pred
abs_errors = np.abs(errors)

# Create DataFrame with results
results_df = pd.DataFrame({
    'actual_brier': y_test,
    'predicted_brier': y_pred,
    'error': errors,
    'abs_error': abs_errors
})

# Add identifiers to the results
results_df = pd.concat([test_ids.reset_index(drop=True), results_df], axis=1)

# Calculate error metrics
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")

# Identify markets with the largest errors
largest_errors = results_df.sort_values('abs_error', ascending=False).head(10)
print("\nMarkets with largest prediction errors:")
print(largest_errors[['question', 'actual_brier', 'predicted_brier', 'error']])

# Save these results
largest_errors.to_csv(f"{output_dir}/largest_error_markets.csv", index=False)

# Visualize error distribution
plt.figure(figsize=(10, 6))
sns.histplot(errors, kde=True)
plt.axvline(x=0, color='r', linestyle='--')
plt.title('Distribution of Prediction Errors')
plt.xlabel('Prediction Error (Actual - Predicted)')
plt.savefig(f"{output_dir}/error_distribution.png", dpi=300)
plt.close()

# Plot actual vs predicted
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.xlabel('Actual Brier Score')
plt.ylabel('Predicted Brier Score')
plt.title('Actual vs Predicted Brier Scores')
plt.savefig(f"{output_dir}/actual_vs_predicted.png", dpi=300)
plt.close()

# Define top features for error analysis
top_features = [
    'price_range', 'unique_traders_count', 'price_fluctuations', 
    'volumeNum', 'final_week_momentum', 'buy_sell_ratio',
    'event_commentCount', 'price_volatility', 'volume_acceleration', 
    'trader_concentration'
]

# Error Analysis by Market Feature
# Load original test data to analyze errors by feature values
X_test_orig = pd.read_csv("modified_analysis/X_test_original.csv")

# Add error to original test data
X_test_with_error = X_test_orig.copy()
X_test_with_error['abs_error'] = abs_errors
X_test_with_error['actual_brier'] = y_test
X_test_with_error['predicted_brier'] = y_pred

# For each top feature, analyze error patterns
for feature in top_features:
    if feature in X_test_with_error.columns:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=feature, y='abs_error', data=X_test_with_error)
        
        # Add trend line
        sns.regplot(x=feature, y='abs_error', data=X_test_with_error, 
                   scatter=False, line_kws={"color": "red"})
        
        plt.title(f'Error Magnitude by {feature}')
        plt.savefig(f"{output_dir}/error_by_{feature}.png", dpi=300)
        plt.close()

# Error patterns table - bin numeric features and check error by bin
error_patterns = []

for feature in top_features:
    if feature in X_test_with_error.columns:
        # Skip if feature has too many unique values (like categorical)
        if X_test_with_error[feature].nunique() > 20 and not pd.api.types.is_categorical_dtype(X_test_with_error[feature]):
            try:
                # Create quartile bins
                X_test_with_error[f'{feature}_bin'] = pd.qcut(X_test_with_error[feature], 4, duplicates='drop')
                
                # Calculate mean error by bin
                bin_errors = X_test_with_error.groupby(f'{feature}_bin')['abs_error'].mean().reset_index()
                
                for _, row in bin_errors.iterrows():
                    error_patterns.append({
                        'feature': feature,
                        'bin': str(row[f'{feature}_bin']),
                        'mean_abs_error': row['abs_error']
                    })
            except Exception as e:
                print(f"Error processing feature {feature}: {e}")

error_patterns_df = pd.DataFrame(error_patterns)
error_patterns_df.to_csv(f"{output_dir}/error_patterns_by_feature_bin.csv", index=False)

print("Prediction error analysis complete!")