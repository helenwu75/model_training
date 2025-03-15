import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline  # Add this import
from imblearn.over_sampling import SMOTE
import joblib

# 1. Load the data
df = pd.read_csv('final_election_results.csv', low_memory=False)
print(f"Original dataset shape: {df.shape}")

# 2. Identify key identifier columns to preserve
id_columns = ['id', 'question', 'slug', 'event_id', 'event_slug']
preserved_identifiers = [col for col in id_columns if col in df.columns]
print(f"Preserving identifiers: {preserved_identifiers}")

# 3. Define target variables
target_variables = ['brier_score', 'log_loss', 'prediction_correct']
available_targets = [col for col in target_variables if col in df.columns]

# 4. Keep only rows where all target variables are not NaN
for target in available_targets:
    df = df[df[target].notna()]

print(f"Dataset shape after removing NaN targets: {df.shape}")

# 5. Identify columns to exclude from modeling (but keep in dataset)
# Text descriptions and dates
exclude_from_model = [col for col in df.columns if col in 
                     ['description', 'startDate', 'endDate', 
                      'market_start_date', 'market_end_date']]

# 6. Check missing values percentage
modeling_cols = [col for col in df.columns if col not in exclude_from_model 
                and col not in preserved_identifiers 
                and col not in available_targets]
                
missing_percentage = df[modeling_cols].isnull().mean() * 100
print("\nMissing values percentage for modeling columns:")
print(missing_percentage[missing_percentage > 0].sort_values(ascending=False))

# 7. Drop columns with too many missing values (e.g., >70%)
keep_for_modeling = missing_percentage[missing_percentage < 70].index.tolist()

# 8. Create two dataframes:
# - complete_df: contains ALL columns including identifiers
# - modeling_df: contains only features for modeling + targets
complete_df = df.copy()
modeling_df = df[keep_for_modeling + available_targets].copy()  # Add .copy() to avoid SettingWithCopyWarning

print(f"\nComplete dataset shape: {complete_df.shape}")
print(f"Modeling dataset shape: {modeling_df.shape}")

# 9. Handle missing values in the modeling features
# Identify numerical and categorical columns
numerical_cols = [col for col in keep_for_modeling if modeling_df[col].dtype.kind in 'fc']
categorical_cols = [col for col in keep_for_modeling if modeling_df[col].dtype == 'object']

# Impute missing values
if numerical_cols:
    imputer = SimpleImputer(strategy='median')
    modeling_df.loc[:, numerical_cols] = imputer.fit_transform(modeling_df[numerical_cols])  # Use .loc to avoid warning

if categorical_cols:
    # For categorical columns, fill with most frequent value
    for col in categorical_cols:
        most_frequent = modeling_df[col].mode().iloc[0] if not modeling_df[col].mode().empty else 'Unknown'
        modeling_df.loc[:, col] = modeling_df[col].fillna(most_frequent)  # Use .loc to avoid warning
        modeling_df.loc[:, col] = modeling_df.loc[:, col].infer_objects(copy=False)


# 10. Check class distribution for prediction_correct
if 'prediction_correct' in modeling_df.columns:
    pred_correct_counts = modeling_df['prediction_correct'].value_counts(normalize=True)
    print(f"\nClass distribution for prediction_correct:\n{pred_correct_counts}")
    minority_class_pct = min(pred_correct_counts) * 100
    print(f"Minority class percentage: {minority_class_pct:.2f}%")

# 11. Split the data into features and targets
X = modeling_df.drop(columns=available_targets)
targets = {}
for target in available_targets:
    targets[target] = modeling_df[target]

# 12. Split the data into train and test sets
X_train, X_test = train_test_split(X, test_size=0.3, random_state=42)

# Get corresponding target splits
y_train_dict = {}
y_test_dict = {}
for target in available_targets:
    y_train_dict[target] = targets[target].loc[X_train.index]
    y_test_dict[target] = targets[target].loc[X_test.index]

print(f"\nTraining set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

# 13. Save indexes for traceability
train_indexes = X_train.index.tolist()
test_indexes = X_test.index.tolist()

# Create traceable dataframes with identifiers
if preserved_identifiers:
    train_ids = complete_df.loc[train_indexes, preserved_identifiers]
    test_ids = complete_df.loc[test_indexes, preserved_identifiers]
    
    # Save these for reference
    train_ids.to_csv('train_identifiers.csv', index=False)
    test_ids.to_csv('test_identifiers.csv', index=False)
    
    print(f"Saved identifiers for {len(train_indexes)} training samples and {len(test_indexes)} test samples")

# 14. Prepare numerical and categorical preprocessors
# Identify numeric and categorical columns in the training data
numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()

# Define preprocessing for numerical and categorical features
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Create preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 15. Apply preprocessing to get a fully numerical representation
print("Applying preprocessing to transform features...")
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

# Get feature names after one-hot encoding
transformed_feature_names = numeric_features.copy()
if categorical_features:
    onehot_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
    categorical_feature_names = onehot_encoder.get_feature_names_out(categorical_features)
    transformed_feature_names = transformed_feature_names + list(categorical_feature_names)

print(f"Number of features after transformation: {len(transformed_feature_names)}")

# 16. Apply SMOTE for classification task if needed
if 'prediction_correct' in available_targets and minority_class_pct < 40:
    print("\nApplying SMOTE to balance classes...")
    
    # Use the preprocessed, fully numerical data for SMOTE
    X_train_for_smote = X_train_preprocessed
    y_train_for_smote = y_train_dict['prediction_correct']
    
    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_for_smote, y_train_for_smote)
    
    # Check class distribution after SMOTE
    smote_class_counts = pd.Series(y_train_smote).value_counts(normalize=True)
    print(f"Class distribution after SMOTE: {smote_class_counts}")
    
    # Store SMOTE-resampled data
    X_train_dict = {'original': X_train_preprocessed, 'smote': X_train_smote}
    y_train_dict['prediction_correct_smote'] = y_train_smote
else:
    print("\nSMOTE not applied (data already balanced or not classification).")
    X_train_dict = {'original': X_train_preprocessed}

# 17. Save the processed data
# Save the cleaned and processed dataframes
complete_df.to_csv('complete_election_data.csv', index=False)
modeling_df.to_csv('cleaned_election_data.csv', index=False)

# Save the original train/test data (before preprocessing) with indexes
X_train.to_csv('X_train_original.csv')
X_test.to_csv('X_test_original.csv')

# Save the preprocessed datasets (fully numerical)
np.save('X_train_preprocessed.npy', X_train_preprocessed)
np.save('X_test_preprocessed.npy', X_test_preprocessed)

# Save the target variables
for target in available_targets:
    pd.DataFrame(y_train_dict[target]).to_csv(f'y_train_{target}.csv')
    pd.DataFrame(y_test_dict[target]).to_csv(f'y_test_{target}.csv')
    np.save(f'y_train_{target}.npy', y_train_dict[target].values)
    np.save(f'y_test_{target}.npy', y_test_dict[target].values)

# Save SMOTE-resampled data if applicable
if 'smote' in X_train_dict:
    np.save('X_train_smote.npy', X_train_dict['smote'])
    np.save('y_train_prediction_correct_smote.npy', y_train_dict['prediction_correct_smote'])

# Save the feature names for later interpretation
pd.Series(transformed_feature_names).to_csv('transformed_feature_names.csv', index=False, header=['feature'])
pd.Series(X_train.columns.tolist()).to_csv('original_feature_names.csv', index=False, header=['feature'])

# Save the preprocessor for future use
joblib.dump(preprocessor, 'feature_preprocessor.joblib')

print("\nData preparation complete. Files saved for modeling.")