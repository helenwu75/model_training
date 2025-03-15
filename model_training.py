import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import joblib
import shap

# 1. Load the preprocessed data
X_train = np.load('X_train_preprocessed.npy')
X_test = np.load('X_test_preprocessed.npy')
y_train_correct = np.load('y_train_prediction_correct.npy')
y_test_correct = np.load('y_test_prediction_correct.npy')

# Load SMOTE-balanced data for classification
X_train_smote = np.load('X_train_smote.npy')
y_train_correct_smote = np.load('y_train_prediction_correct_smote.npy')

# Load feature names for interpretation
feature_names = pd.read_csv('transformed_feature_names.csv')['feature'].tolist()

# 2. Define and train the four classification models
print("Training classification models...")

# Model 1: Random Forest
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train_smote, y_train_correct_smote)

# Model 2: Logistic Regression with L1 regularization
# Using C=0.1 for stronger regularization, can be tuned based on results
lr_l1_model = LogisticRegression(penalty='l1', solver='liblinear', C=0.1, random_state=42)
lr_l1_model.fit(X_train_smote, y_train_correct_smote)

# Model 3: Gradient Boosting
gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
gb_model.fit(X_train_smote, y_train_correct_smote)

# Model 4: PCA + Logistic Regression
# Using PCA to reduce to approximately 95% explained variance
pca_lr_pipeline = Pipeline([
    ('pca', PCA(n_components=0.95, random_state=42)),
    ('lr', LogisticRegression(random_state=42))
])
pca_lr_pipeline.fit(X_train_smote, y_train_correct_smote)

# Store models in a dictionary for evaluation
models = {
    'Random Forest': rf_model,
    'Logistic Regression (L1)': lr_l1_model,
    'Gradient Boosting': gb_model,
    'PCA + Logistic Regression': pca_lr_pipeline
}

# 3. Evaluate all models
results = {}
for name, model in models.items():
    # Predict on test set
    y_pred = model.predict(X_test)
    
    # Calculate probabilities for ROC and PR curves
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        # For pipeline or other models
        y_proba = model.predict_proba(X_test)[:, 1]
    
    # Compute metrics
    accuracy = accuracy_score(y_test_correct, y_pred)
    conf_matrix = confusion_matrix(y_test_correct, y_pred)
    report = classification_report(y_test_correct, y_pred, output_dict=True)
    roc_auc = roc_auc_score(y_test_correct, y_proba)
    avg_precision = average_precision_score(y_test_correct, y_proba)
    
    # Store results
    results[name] = {
        'model': model,
        'accuracy': accuracy,
        'confusion_matrix': conf_matrix,
        'classification_report': report,
        'roc_auc': roc_auc,
        'avg_precision': avg_precision,
        'y_pred': y_pred,
        'y_proba': y_proba
    }
    
    print(f"\n{name} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"Average Precision: {avg_precision:.4f}")
    print(f"Confusion Matrix:\n{conf_matrix}")
    
    # Show classification report
    print("Classification Report:")
    cls_report = pd.DataFrame(report).T
    print(cls_report)

# 4. Compare models
# Create a comparison dataframe
comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Accuracy': [results[model]['accuracy'] for model in results],
    'ROC AUC': [results[model]['roc_auc'] for model in results],
    'Avg Precision': [results[model]['avg_precision'] for model in results]
})

# Add F1 scores - handle both string and numeric keys
for model in results:
    report = results[model]['classification_report']
    # Check what keys are in the report (excluding 'accuracy', 'macro avg', etc.)
    class_keys = [key for key in report.keys() if key not in ['accuracy', 'macro avg', 'weighted avg']]
    
    # Sort class keys if they're numeric
    try:
        class_keys = sorted([float(k) for k in class_keys])
        class_keys = [str(k) for k in class_keys]  # Convert back to strings
    except:
        # If conversion fails, keep them as they are
        pass
    
    # Assuming binary classification, with class_keys[0] as negative and class_keys[1] as positive
    if len(class_keys) >= 2:
        comparison_df.loc[comparison_df['Model'] == model, 'F1 (Incorrect)'] = report[class_keys[0]]['f1-score']
        comparison_df.loc[comparison_df['Model'] == model, 'F1 (Correct)'] = report[class_keys[1]]['f1-score']
    else:
        # Fallback if we don't have two classes
        comparison_df.loc[comparison_df['Model'] == model, 'F1 (Incorrect)'] = np.nan
        comparison_df.loc[comparison_df['Model'] == model, 'F1 (Correct)'] = np.nan

# 5. Plot ROC curves for all models
plt.figure(figsize=(10, 8))
for name in results:
    fpr, tpr, _ = roc_curve(y_test_correct, results[name]['y_proba'])
    plt.plot(fpr, tpr, label=f"{name} (AUC = {results[name]['roc_auc']:.4f})")

plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Classification Models')
plt.legend()
plt.savefig('model_comparison_roc_curves.png')

# 6. Feature importance analysis
# 6.1 Random Forest feature importance
if 'Random Forest' in results:
    rf_importances = pd.DataFrame({
        'feature': feature_names,
        'importance': results['Random Forest']['model'].feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=rf_importances.head(20))
    plt.title('Top 20 Features - Random Forest')
    plt.tight_layout()
    plt.savefig('rf_feature_importance.png')
    
    # Save importance scores
    rf_importances.to_csv('rf_feature_importance.csv', index=False)

# 6.2 Logistic Regression coefficients (for L1 model)
if 'Logistic Regression (L1)' in results:
    # Get non-zero coefficients for L1 regression
    coef = results['Logistic Regression (L1)']['model'].coef_[0]
    l1_importance = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coef
    })
    
    # Sort by absolute coefficient value
    l1_importance['abs_coef'] = l1_importance['coefficient'].abs()
    l1_importance = l1_importance.sort_values('abs_coef', ascending=False)
    
    # Plot top coefficients
    plt.figure(figsize=(12, 8))
    top_coefs = l1_importance.head(20)
    sns.barplot(x='coefficient', y='feature', data=top_coefs)
    plt.title('Top 20 Features - Logistic Regression (L1)')
    plt.tight_layout()
    plt.savefig('l1_logistic_coefficients.png')
    
    # Save coefficients
    l1_importance.to_csv('l1_logistic_coefficients.csv', index=False)

# 6.3 Gradient Boosting feature importance
if 'Gradient Boosting' in results:
    gb_importances = pd.DataFrame({
        'feature': feature_names,
        'importance': results['Gradient Boosting']['model'].feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=gb_importances.head(20))
    plt.title('Top 20 Features - Gradient Boosting')
    plt.tight_layout()
    plt.savefig('gb_feature_importance.png')
    
    gb_importances.to_csv('gb_feature_importance.csv', index=False)

# 6.4 PCA Component Analysis (for PCA+LR model)
if 'PCA + Logistic Regression' in results:
    pca = results['PCA + Logistic Regression']['model'].named_steps['pca']
    
    # Get explained variance ratio
    explained_variance = pca.explained_variance_ratio_
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(explained_variance) + 1), explained_variance)
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('PCA Explained Variance')
    plt.savefig('pca_explained_variance.png')
    
    # Analyze component loadings for top components
    n_components = min(5, pca.n_components_)  # Show top 5 or fewer
    loadings = pd.DataFrame(
        data=pca.components_[:n_components, :],
        columns=feature_names
    )
    
    # For each component, get top features by absolute loading
    top_loadings = []
    for i in range(n_components):
        component_loadings = pd.DataFrame({
            'feature': feature_names,
            'loading': loadings.iloc[i, :],
            'abs_loading': abs(loadings.iloc[i, :])
        }).sort_values('abs_loading', ascending=False)
        
        top_loadings.append(component_loadings.head(10).copy())
        top_loadings[-1]['component'] = f"PC{i+1}"
    
    # Combine all components' top loadings
    all_top_loadings = pd.concat(top_loadings)
    all_top_loadings.to_csv('pca_top_loadings.csv', index=False)

# 7. SHAP analysis for best model
# Determine best model based on ROC AUC
best_model_name = comparison_df.iloc[0]['Model']
best_model = results[best_model_name]['model']

print(f"\nPerforming SHAP analysis for best model: {best_model_name}")

# For tree-based models
if best_model_name in ['Random Forest', 'Gradient Boosting']:
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_test)
    
    # If shap_values is a list (for multi-class), get the values for class 1
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, max_display=20)
    plt.savefig(f'shap_summary_{best_model_name.replace(" ", "_").lower()}.png')
    
    # SHAP force plot for a few examples
    plt.figure(figsize=(20, 3))
    shap.initjs()
    force_plot = shap.force_plot(explainer.expected_value, 
                              shap_values[:5], 
                              X_test[:5], 
                              feature_names=feature_names)
    shap.save_html(f'shap_force_plot_{best_model_name.replace(" ", "_").lower()}.html', force_plot)
    
else:
    # For linear models or pipelines
    explainer = shap.KernelExplainer(best_model.predict_proba, 
                                   shap.kmeans(X_train_smote, 50))
    shap_values = explainer.shap_values(X_test[:100])  # Using subset for computational efficiency
    
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values[1], X_test[:100], feature_names=feature_names, max_display=20)
    plt.savefig(f'shap_summary_{best_model_name.replace(" ", "_").lower()}.png')

# 8. Save all models
for name, model_dict in results.items():
    model = model_dict['model']
    joblib.dump(model, f'{name.replace(" ", "_").lower()}_model.joblib')

print("\nModel training, evaluation, and feature importance analysis complete!")