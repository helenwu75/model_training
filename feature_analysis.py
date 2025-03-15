"""
Feature Importance Analysis for Polymarket Election Markets

This script performs a comprehensive feature importance analysis for Polymarket
election market data, using three different models:
1. Gradient Boosting Regressor
2. Random Forest Regressor
3. Linear Regression with L1 Regularization (Lasso)

The analysis includes:
- Model-specific importance
- Permutation importance (model-agnostic)
- SHAP values for the best-performing model
- Consensus feature ranking

Author: UChicago ML Professor
Date: March 15, 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from datetime import datetime
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
import shap
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set up directories
INPUT_DIR = "modified_analysis"
OUTPUT_DIR = "feature_importance_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Results will be saved to: {OUTPUT_DIR}")

# Create a timestamp for this run
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
print(f"Analysis started at: {timestamp}")

# 1. Load the preprocessed data
print("Loading preprocessed data...")
try:
    X_train = np.load(os.path.join(INPUT_DIR, 'X_train_preprocessed.npy'))
    X_test = np.load(os.path.join(INPUT_DIR, 'X_test_preprocessed.npy'))
    
    # Load Brier score target (primary)
    y_train_brier = np.load(os.path.join(INPUT_DIR, 'y_train_brier_score.npy'))
    y_test_brier = np.load(os.path.join(INPUT_DIR, 'y_test_brier_score.npy'))
    
    # Load log loss target (secondary)
    y_train_logloss = np.load(os.path.join(INPUT_DIR, 'y_train_log_loss.npy'))
    y_test_logloss = np.load(os.path.join(INPUT_DIR, 'y_test_log_loss.npy'))
    
    # Load binary target (for reference)
    y_train_binary = np.load(os.path.join(INPUT_DIR, 'y_train_prediction_correct.npy'))
    y_test_binary = np.load(os.path.join(INPUT_DIR, 'y_test_prediction_correct.npy'))
    
    # Load feature names
    feature_names_path = os.path.join(INPUT_DIR, 'transformed_feature_names.csv')
    feature_names = pd.read_csv(feature_names_path)['feature'].tolist()
    
    print(f"Loaded data with {X_train.shape[1]} features, {len(y_train_brier)} training samples, and {len(y_test_brier)} test samples.")
    
except Exception as e:
    print(f"Error loading data: {e}")
    print("Please make sure the preprocessed files exist in the 'modified_analysis' directory.")
    exit(1)

# 2. Define and train regression models
print("\nTraining regression models to predict Brier score...")

# Dictionary to store models and their parameters
models_config = {
    'Gradient Boosting': {
        'model': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42),
        'params': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 5}
    },
    'Random Forest': {
        'model': RandomForestRegressor(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42),
        'params': {'n_estimators': 100, 'max_depth': 10}
    },
    'Lasso Regression': {
        'model': Lasso(alpha=0.01, max_iter=10000, random_state=42),
        'params': {'alpha': 0.01, 'max_iter': 10000}
    }
}

# Train all models and collect results
results = {}
for name, config in models_config.items():
    print(f"\nTraining {name}...")
    model = config['model']
    
    # Train model on Brier score
    model.fit(X_train, y_train_brier)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test_brier, y_pred))
    mae = mean_absolute_error(y_test_brier, y_pred)
    r2 = r2_score(y_test_brier, y_pred)
    
    # Store results
    results[name] = {
        'model': model,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'y_pred': y_pred
    }
    
    print(f"  RMSE: {rmse:.6f}")
    print(f"  MAE: {mae:.6f}")
    print(f"  R²: {r2:.6f}")

# 3. Compare model performance
comparison_data = [
    {
        'Model': name,
        'RMSE': result['rmse'],
        'MAE': result['mae'],
        'R²': result['r2']
    }
    for name, result in results.items()
]

comparison_df = pd.DataFrame(comparison_data)
comparison_df = comparison_df.sort_values('RMSE', ascending=True)

print("\nModel Performance Comparison:")
print(comparison_df)

# Save to CSV
comparison_path = os.path.join(OUTPUT_DIR, 'model_performance_comparison.csv')
comparison_df.to_csv(comparison_path, index=False)
print(f"Model comparison saved to {comparison_path}")

# Determine best model based on RMSE
best_model_name = comparison_df.iloc[0]['Model']
print(f"\nBest model based on RMSE: {best_model_name}")

# 4. Extract model-specific feature importance
importance_methods = {}

def analyze_rf_importance(model, feature_names):
    """Extract feature importance from Random Forest model"""
    if not isinstance(model, RandomForestRegressor):
        print("Model is not a Random Forest. Skipping this analysis.")
        return None
    
    print("\nExtracting Random Forest feature importance...")
    rf_importances = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=rf_importances.head(20))
    plt.title('Top 20 Features - Random Forest')
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'rf_feature_importance.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    # Save importance scores
    rf_importances.to_csv(os.path.join(OUTPUT_DIR, 'rf_feature_importance.csv'), index=False)
    print(f"Random Forest feature importance saved to {OUTPUT_DIR}")
    
    return rf_importances

def analyze_gb_importance(model, feature_names):
    """Extract feature importance from Gradient Boosting model"""
    if not isinstance(model, GradientBoostingRegressor):
        print("Model is not a Gradient Boosting Regressor. Skipping this analysis.")
        return None
    
    print("\nExtracting Gradient Boosting feature importance...")
    gb_importances = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=gb_importances.head(20))
    plt.title('Top 20 Features - Gradient Boosting')
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'gb_feature_importance.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    # Save importance scores
    gb_importances.to_csv(os.path.join(OUTPUT_DIR, 'gb_feature_importance.csv'), index=False)
    print(f"Gradient Boosting feature importance saved to {OUTPUT_DIR}")
    
    return gb_importances

def analyze_lasso_coefficients(model, feature_names):
    """Extract coefficients from Lasso regression model"""
    if not isinstance(model, Lasso):
        print("Model is not a Lasso Regression. Skipping this analysis.")
        return None
    
    print("\nExtracting Lasso regression coefficients...")
    coef = model.coef_
    lasso_importance = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coef
    })
    
    # Sort by absolute coefficient value
    lasso_importance['abs_coef'] = lasso_importance['coefficient'].abs()
    lasso_importance = lasso_importance.sort_values('abs_coef', ascending=False)
    
    # Count non-zero coefficients
    non_zero = (lasso_importance['coefficient'] != 0).sum()
    print(f"Lasso selected {non_zero} features out of {len(feature_names)}")
    
    # Plot top coefficients
    plt.figure(figsize=(12, 8))
    top_coefs = lasso_importance.head(20)
    sns.barplot(x='coefficient', y='feature', data=top_coefs)
    plt.title('Top 20 Features - Lasso Regression')
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'lasso_coefficients.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    # Save coefficients
    lasso_importance.to_csv(os.path.join(OUTPUT_DIR, 'lasso_coefficients.csv'), index=False)
    print(f"Lasso regression coefficients saved to {OUTPUT_DIR}")
    
    return lasso_importance

# Extract feature importance for each model
if 'Random Forest' in results:
    rf_importance = analyze_rf_importance(results['Random Forest']['model'], feature_names)
    importance_methods['Random Forest'] = rf_importance

if 'Gradient Boosting' in results:
    gb_importance = analyze_gb_importance(results['Gradient Boosting']['model'], feature_names)
    importance_methods['Gradient Boosting'] = gb_importance

if 'Lasso Regression' in results:
    lasso_importance = analyze_lasso_coefficients(results['Lasso Regression']['model'], feature_names)
    # For lasso, use absolute coefficient value
    if lasso_importance is not None:
        lasso_imp = lasso_importance.copy()
        lasso_imp = lasso_imp[['feature', 'abs_coef']].rename(columns={'abs_coef': 'importance'})
        importance_methods['Lasso Regression'] = lasso_imp

# 5. Calculate permutation importance (model-agnostic)
print("\nCalculating permutation importance for all models...")

permutation_results = {}
for name, result in results.items():
    print(f"Calculating permutation importance for {name}...")
    model = result['model']
    
    # Calculate permutation importance on test set
    perm_importance = permutation_importance(
        model, X_test, y_test_brier, 
        n_repeats=10, 
        random_state=42,
        n_jobs=-1
    )
    
    # Create dataframe
    perm_imp_df = pd.DataFrame({
        'feature': feature_names,
        'importance': perm_importance.importances_mean
    }).sort_values('importance', ascending=False)
    
    # Save to CSV
    perm_imp_df.to_csv(os.path.join(OUTPUT_DIR, f'permutation_importance_{name.replace(" ", "_").lower()}.csv'), index=False)
    
    # Plot top features
    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=perm_imp_df.head(20))
    plt.title(f'Top 20 Features - Permutation Importance ({name})')
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, f'permutation_importance_{name.replace(" ", "_").lower()}.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    # Store for later use
    permutation_results[name] = perm_imp_df
    importance_methods[f'{name} (Permutation)'] = perm_imp_df
    
    print(f"Permutation importance for {name} saved to {OUTPUT_DIR}")

# 6. Perform SHAP analysis for best model
best_model = results[best_model_name]['model']

print(f"\nPerforming SHAP analysis for best model ({best_model_name})...")

shap_values = None
try:
    # Choose the appropriate explainer based on the model type
    if isinstance(best_model, (RandomForestRegressor, GradientBoostingRegressor)):
        # Use TreeExplainer for tree-based models
        explainer = shap.TreeExplainer(best_model)
        
        # Calculate SHAP values on a subset of test data for efficiency
        sample_size = min(100, X_test.shape[0])
        X_test_sample = X_test[:sample_size]
        shap_values = explainer.shap_values(X_test_sample)
        
        # Create summary plot
        plt.figure(figsize=(12, 10))
        shap.summary_plot(
            shap_values, 
            X_test_sample, 
            feature_names=feature_names, 
            max_display=20,
            show=False
        )
        plt.tight_layout()
        output_path = os.path.join(OUTPUT_DIR, f'shap_summary_{best_model_name.replace(" ", "_").lower()}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create feature importance based on SHAP values
        shap_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': np.abs(shap_values).mean(0)
        }).sort_values('importance', ascending=False)
        
        # Save to CSV
        shap_importance.to_csv(os.path.join(OUTPUT_DIR, f'shap_importance_{best_model_name.replace(" ", "_").lower()}.csv'), index=False)
        importance_methods['SHAP'] = shap_importance
        
    elif isinstance(best_model, Lasso):
        # For linear models, we can use LinearExplainer
        # Create a background dataset with kmeans
        train_summary = shap.kmeans(X_train, k=min(50, X_train.shape[0]))
        explainer = shap.LinearExplainer(best_model, train_summary)
        
        # Calculate SHAP values on a subset of test data for efficiency
        sample_size = min(100, X_test.shape[0])
        X_test_sample = X_test[:sample_size]
        shap_values = explainer.shap_values(X_test_sample)
        
        # Create summary plot
        plt.figure(figsize=(12, 10))
        shap.summary_plot(
            shap_values, 
            X_test_sample, 
            feature_names=feature_names, 
            max_display=20,
            show=False
        )
        plt.tight_layout()
        output_path = os.path.join(OUTPUT_DIR, f'shap_summary_{best_model_name.replace(" ", "_").lower()}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create feature importance based on SHAP values
        shap_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': np.abs(shap_values).mean(0)
        }).sort_values('importance', ascending=False)
        
        # Save to CSV
        shap_importance.to_csv(os.path.join(OUTPUT_DIR, f'shap_importance_{best_model_name.replace(" ", "_").lower()}.csv'), index=False)
        importance_methods['SHAP'] = shap_importance
    
    print(f"SHAP analysis for {best_model_name} saved to {OUTPUT_DIR}")
    
    # Create SHAP dependence plots for top 5 features
    if shap_values is not None:
        top_features = shap_importance.head(5)['feature'].tolist()
        
        for i, feature in enumerate(top_features):
            feature_idx = feature_names.index(feature)
            plt.figure(figsize=(10, 7))
            shap.dependence_plot(
                feature_idx,
                shap_values,
                X_test_sample,
                feature_names=feature_names,
                show=False
            )
            plt.title(f'SHAP Dependence Plot for {feature}')
            plt.tight_layout()
            output_path = os.path.join(OUTPUT_DIR, f'shap_dependence_{feature.replace(" ", "_").lower()}.png')
            plt.savefig(output_path, dpi=300)
            plt.close()
            
        print(f"SHAP dependence plots for top 5 features saved to {OUTPUT_DIR}")

except Exception as e:
    print(f"Error in SHAP analysis: {e}")
    import traceback
    traceback.print_exc()

# 7. Create consensus feature ranking
if len(importance_methods) > 1:
    print("\nCreating consensus feature ranking across all methods...")
    
    # Get top 15 features from each method
    top_features = {}
    for method, imp_df in importance_methods.items():
        top_features[method] = imp_df.head(15)['feature'].tolist()
    
    # Find features that appear in multiple methods
    all_top_features = []
    for method, features in top_features.items():
        all_top_features.extend(features)
    
    # Count frequency of each feature
    feature_counts = Counter(all_top_features)
    
    # Features that appear in multiple methods
    common_features = [feature for feature, count in feature_counts.items() if count > 1]
    
    print(f"Features important across multiple methods: {len(common_features)}")
    for feature in common_features:
        methods = [method for method, features in top_features.items() if feature in features]
        print(f"  {feature}: found in {len(methods)} methods - {', '.join(methods)}")
    
    # Create a ranking based on frequency and average rank
    rankings = {}
    for method, imp_df in importance_methods.items():
        # Get ranks (1-based)
        imp_df['rank'] = imp_df['importance'].rank(ascending=False)
        # Create a mapping from feature to rank
        rankings[method] = dict(zip(imp_df['feature'], imp_df['rank']))
    
    # Create comprehensive ranking for all features
    all_features = list(set(feature_names))
    rank_rows = []
    
    for feature in all_features:
        row = {'feature': feature}
        
        # Count in how many methods this feature appears in top 15
        methods_count = sum(1 for method, top in top_features.items() if feature in top)
        row['top15_count'] = methods_count
        
        # Get average rank across all methods (only where feature is ranked)
        feature_ranks = []
        for method, rank_dict in rankings.items():
            if feature in rank_dict:
                feature_ranks.append(rank_dict[feature])
        
        if feature_ranks:
            row['avg_rank'] = sum(feature_ranks) / len(feature_ranks)
        else:
            row['avg_rank'] = float('inf')  # Not ranked in any method
        
        # Add individual method ranks
        for method in rankings:
            row[f"{method}_rank"] = rankings[method].get(feature, np.nan)
        
        rank_rows.append(row)
    
    # Create dataframe with all rankings
    all_rankings = pd.DataFrame(rank_rows)
    
    # Sort by count in top 15, then by average rank
    all_rankings = all_rankings.sort_values(
        ['top15_count', 'avg_rank'], 
        ascending=[False, True]
    )
    
    # Save full rankings
    all_rankings.to_csv(os.path.join(OUTPUT_DIR, 'consensus_feature_ranking_full.csv'), index=False)
    
    # Create a simplified consensus ranking with just the top features
    top_consensus = all_rankings[all_rankings['top15_count'] > 1].head(20)
    top_consensus.to_csv(os.path.join(OUTPUT_DIR, 'consensus_feature_ranking_top20.csv'), index=False)
    
    # Create visualization of consensus top 15
    plt.figure(figsize=(12, 10))
    consensus_vis = top_consensus.head(15).copy()
    
    # Create a score for visualization (inverse of average rank, scaled to 0-1)
    max_rank = consensus_vis['avg_rank'].max()
    consensus_vis['consensus_score'] = 1 - (consensus_vis['avg_rank'] / max_rank)
    
    # Create color mapping based on appearance count
    cmap = plt.cm.get_cmap('viridis', consensus_vis['top15_count'].max() + 1)
    colors = [cmap(count) for count in consensus_vis['top15_count']]
    
    # Create a horizontal bar chart
    plt.barh(
        y=range(len(consensus_vis)),
        width=consensus_vis['consensus_score'],
        color=colors,
        alpha=0.8
    )
    
    # Add text labels showing count and average rank
    for i, (_, row) in enumerate(consensus_vis.iterrows()):
        plt.text(
            row['consensus_score'] + 0.01,
            i,
            f"Count: {row['top15_count']}, Avg Rank: {row['avg_rank']:.1f}",
            va='center',
            fontsize=10
        )
    
    # Set y-tick labels to feature names
    plt.yticks(range(len(consensus_vis)), consensus_vis['feature'])
    plt.xlabel('Consensus Score')
    plt.title('Top 15 Features by Consensus Ranking')
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'consensus_feature_ranking_top15.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"Consensus feature ranking saved to {OUTPUT_DIR}")
    
    # 8. Create a consolidated feature importance visualization for the top 10 consensus features
    top10_features = all_rankings.head(10)['feature'].tolist()
    print("\nCreating consolidated importance visualization for top 10 consensus features...")
    
    # Get importance values for each method and feature
    consolidated_data = []
    
    for feature in top10_features:
        for method, imp_df in importance_methods.items():
            if feature in imp_df['feature'].values:
                importance_value = imp_df.loc[imp_df['feature'] == feature, 'importance'].values[0]
                consolidated_data.append({
                    'feature': feature,
                    'method': method,
                    'importance': importance_value
                })
    
    if consolidated_data:
        consolidated_df = pd.DataFrame(consolidated_data)
        
        # Normalize importance values within each method (0-1 scale)
        methods = consolidated_df['method'].unique()
        for method in methods:
            method_mask = consolidated_df['method'] == method
            method_max = consolidated_df.loc[method_mask, 'importance'].max()
            if method_max > 0:  # Avoid division by zero
                consolidated_df.loc[method_mask, 'importance_normalized'] = (
                    consolidated_df.loc[method_mask, 'importance'] / method_max
                )
            else:
                consolidated_df.loc[method_mask, 'importance_normalized'] = 0
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        pivot_df = consolidated_df.pivot(index='feature', columns='method', values='importance_normalized')
        sns.heatmap(pivot_df, annot=True, cmap='viridis', fmt='.2f')
        plt.title('Normalized Feature Importance Across Methods (Top 10 Consensus Features)')
        plt.tight_layout()
        output_path = os.path.join(OUTPUT_DIR, 'consolidated_feature_importance_heatmap.png')
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        print(f"Consolidated feature importance visualization saved to {OUTPUT_DIR}")

# 9. Show interpretable insights for top features
print("\nGenerating interpretable insights for top features...")

try:
    # Load original data to understand feature values
    X_original_df = pd.read_csv(os.path.join(INPUT_DIR, 'X_train_original.csv'), index_col=0)
    
    # Get top 10 consensus features if available, otherwise use best model's top features
    if 'top10_features' in locals():
        top_interpret_features = top10_features
    else:
        # Use best model's top features
        model_importance = importance_methods.get(best_model_name)
        if model_importance is not None:
            top_interpret_features = model_importance.head(10)['feature'].tolist()
        else:
            # Fallback to first available importance method
            method_name = list(importance_methods.keys())[0]
            top_interpret_features = importance_methods[method_name].head(10)['feature'].tolist()
    
    # Filter to features actually in the original dataset
    available_features = [f for f in top_interpret_features if f in X_original_df.columns]
    
    if available_features:
        # Calculate statistics for these features
        feature_stats = []
        
        for feature in available_features:
            # Basic statistics
            feature_data = X_original_df[feature]
            
            if pd.api.types.is_numeric_dtype(feature_data):
                stats = {
                    'feature': feature,
                    'mean': feature_data.mean(),
                    'median': feature_data.median(),
                    'std': feature_data.std(),
                    'min': feature_data.min(),
                    'max': feature_data.max(),
                    'type': 'numeric'
                }
                
                # Add correlation with target
                if 'y_train_brier' in locals():
                    target_series = pd.Series(y_train_brier, index=X_original_df.index)
                    corr = feature_data.corr(target_series)
                    stats['correlation_with_brier'] = corr
                
                feature_stats.append(stats)
                
            elif pd.api.types.is_categorical_dtype(feature_data) or feature.startswith('event_'):
                # For categorical features, get value counts
                stats = {
                    'feature': feature,
                    'unique_values': feature_data.nunique(),
                    'type': 'categorical'
                }
                
                # Get value distribution
                value_counts = feature_data.value_counts(normalize=True).to_dict()
                top_values = sorted(value_counts.items(), key=lambda x: x[1], reverse=True)[:3]
                stats['top_values'] = str(top_values)
                
                feature_stats.append(stats)
        
        # Create interpretable insights dataframe
        interpret_df = pd.DataFrame(feature_stats)
        interpret_df.to_csv(os.path.join(OUTPUT_DIR, 'feature_interpretation.csv'), index=False)
        
        print(f"Feature interpretation saved to {OUTPUT_DIR}")
        
        # Print a summary of feature insights
        print("\nSummary of Top Features:")
        for _, row in interpret_df.iterrows():
            feature = row['feature']
            
            if row['type'] == 'numeric':
                print(f"  {feature}: Mean={row['mean']:.4f}, Median={row['median']:.4f}, Range=[{row['min']:.4f}, {row['max']:.4f}]")
                if 'correlation_with_brier' in row:
                    corr = row['correlation_with_brier']
                    direction = "HIGHER" if corr > 0 else "LOWER"
                    print(f"    Correlation with Brier score: {corr:.4f} ({direction} values → worse predictions)")
            else:
                print(f"  {feature}: Categorical with {row['unique_values']} unique values")
                print(f"    Top values: {row['top_values']}")
    
    else:
        print("No top features found in original dataset for interpretation")

except Exception as e:
    print(f"Error generating interpretable insights: {e}")
    import traceback
    traceback.print_exc()

# 10. Save all models
for name, result in results.items():
    model = result['model']
    model_path = os.path.join(OUTPUT_DIR, f"{name.replace(' ', '_').lower()}_model.joblib")
    joblib.dump(model, model_path)
    print(f"Saved {name} model to {model_path}")

print("\nFeature importance analysis complete!")
print(f"All results saved to {OUTPUT_DIR}")