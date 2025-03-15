import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import joblib

# Create output directory for the modified files
OUTPUT_DIR = "modified_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Output files will be saved to: {OUTPUT_DIR}")

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

# 5. Exclude features that would cause leakage or aren't relevant for modeling
# Features that directly relate to the prediction outcome
leakage_features = [
    'outcomePrices', 'correct_outcome', 'actual_outcome',  # Actual outcomes
    'closing_price', 'last_trade_price',  # Direct price inputs to prediction
    'prediction_error', 'prediction_confidence',  # Derived from the target
    'price_2days_prior', 'pre_election_vwap_48h'  # Could have direct relationship with outcome
]

# Text descriptions, dates, and technical details
exclude_from_model = [
    'description', 'startDate', 'endDate', 'market_start_date', 'market_end_date',
    'groupItemTitle', 'outcomes', 'clobTokenIds', 'yes_token_id',
    'enableOrderBook', 'active', 'event_ticker', 'event_title', 'event_description'
]

# Combine all exclusions
all_exclusions = leakage_features + exclude_from_model

# 6. Define the desired features (as specified in your requirements)
desired_features = [
    'volumeNum', 'event_country', 'event_electionType', 'event_commentCount',
    'price_volatility', 'price_range', 'final_week_momentum', 'price_fluctuations',
    'market_duration_days', 'trading_frequency', 'buy_sell_ratio', 'trading_continuity',
    'late_stage_participation', 'volume_acceleration', 'unique_traders_count',
    'trader_to_trade_ratio', 'two_way_traders_ratio', 'trader_concentration',
    'new_trader_influx', 'comment_per_vol', 'comment_per_trader'
]

# Filter to only include desired features that are in the dataset
modeling_features = [feature for feature in desired_features if feature in df.columns]
print(f"\nSelected modeling features: {len(modeling_features)}")
print(modeling_features)

# 7. Check missing values percentage
missing_percentage = df[modeling_features].isnull().mean() * 100
print("\nMissing values percentage for modeling features:")
print(missing_percentage[missing_percentage > 0].sort_values(ascending=False))

# 8. Drop features with too many missing values (e.g., >70%)
features_to_keep = missing_percentage[missing_percentage < 70].index.tolist()
print(f"\nFeatures kept after missing value check: {len(features_to_keep)}")

if len(features_to_keep) < len(modeling_features):
    print("Dropped features due to missing values:")
    print([f for f in modeling_features if f not in features_to_keep])

# 9. Create two dataframes:
# - complete_df: contains ALL columns including identifiers
# - modeling_df: contains only features for modeling + targets
complete_df = df.copy()
modeling_df = df[features_to_keep + available_targets].copy()

print(f"\nComplete dataset shape: {complete_df.shape}")
print(f"Modeling dataset shape: {modeling_df.shape}")

# 10. Handle missing values in the modeling features
# Identify numerical and categorical columns
numerical_cols = [col for col in features_to_keep if modeling_df[col].dtype.kind in 'fc']
categorical_cols = [col for col in features_to_keep if modeling_df[col].dtype == 'object']

print(f"\nNumerical features: {len(numerical_cols)}")
print(f"Categorical features: {len(categorical_cols)}")

# Impute missing values
if numerical_cols:
    imputer = SimpleImputer(strategy='median')
    modeling_df.loc[:, numerical_cols] = imputer.fit_transform(modeling_df[numerical_cols])

if categorical_cols:
    # For categorical columns, fill with most frequent value
    for col in categorical_cols:
        most_frequent = modeling_df[col].mode().iloc[0] if not modeling_df[col].mode().empty else 'Unknown'
        modeling_df.loc[:, col] = modeling_df[col].fillna(most_frequent)
        modeling_df.loc[:, col] = modeling_df.loc[:, col].infer_objects(copy=False)

# 11. Check class distribution for prediction_correct
if 'prediction_correct' in modeling_df.columns:
    pred_correct_counts = modeling_df['prediction_correct'].value_counts(normalize=True)
    print(f"\nClass distribution for prediction_correct:\n{pred_correct_counts}")
    minority_class_pct = min(pred_correct_counts) * 100
    print(f"Minority class percentage: {minority_class_pct:.2f}%")

# 12. Split the data into features and targets
X = modeling_df.drop(columns=available_targets)
targets = {}
for target in available_targets:
    targets[target] = modeling_df[target]

# 13. Split the data into train and test sets
# Use stratified split for classification task
if 'prediction_correct' in available_targets:
    X_train, X_test, y_train_pred, y_test_pred = train_test_split(
        X, targets['prediction_correct'], 
        test_size=0.3, 
        random_state=42,
        stratify=targets['prediction_correct']  # Ensure balanced classes in both splits
    )
    
    # Reconstruct the train/test index sets
    train_indexes = X_train.index
    test_indexes = X_test.index
    
    # Get other targets using these indexes
    y_train_dict = {'prediction_correct': y_train_pred}
    y_test_dict = {'prediction_correct': y_test_pred}
    
    for target in available_targets:
        if target != 'prediction_correct':
            y_train_dict[target] = targets[target].loc[train_indexes]
            y_test_dict[target] = targets[target].loc[test_indexes]
else:
    # For regression tasks, use regular split
    X_train, X_test = train_test_split(X, test_size=0.3, random_state=42)
    
    # Get corresponding target splits
    train_indexes = X_train.index
    test_indexes = X_test.index
    
    y_train_dict = {}
    y_test_dict = {}
    for target in available_targets:
        y_train_dict[target] = targets[target].loc[train_indexes]
        y_test_dict[target] = targets[target].loc[test_indexes]

print(f"\nTraining set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

# 14. Save indexes for traceability
train_indexes_list = train_indexes.tolist()
test_indexes_list = test_indexes.tolist()

# Create traceable dataframes with identifiers
if preserved_identifiers:
    train_ids = complete_df.loc[train_indexes_list, preserved_identifiers]
    test_ids = complete_df.loc[test_indexes_list, preserved_identifiers]
    
    # Save these for reference
    train_ids.to_csv(os.path.join(OUTPUT_DIR, 'train_identifiers.csv'), index=False)
    test_ids.to_csv(os.path.join(OUTPUT_DIR, 'test_identifiers.csv'), index=False)
    
    print(f"Saved identifiers for {len(train_indexes_list)} training samples and {len(test_indexes_list)} test samples")

# 15. Prepare numerical and categorical preprocessors

# Explicitly define which columns are categorical vs numerical
# This prevents numeric comment features from being treated as categorical
categorical_features_raw = ['event_country', 'event_electionType']
categorical_features = [col for col in categorical_features_raw if col in X_train.columns]
numeric_features = [col for col in X_train.columns if col not in categorical_features]

print(f"\nExplicitly categorized features:")
print(f"  Categorical: {categorical_features}")
print(f"  Numerical: {numeric_features}")

# Group rare categories in election type
if 'event_electionType' in X_train.columns:
    print("\nGrouping rare election types...")
    
    # Define election type groupings
    regional_elections = ['Provincial', 'Governor', 'Mayoral']
    presidential_details = ['Presidential Popular Vote', 'Presidential Tipping Point', 
                           'Electoral College', 'Presidential Speech']
    
    # Apply groupings to both train and test sets
    for df in [X_train, X_test]:
        # Remove leading/trailing spaces
        df['event_electionType'] = df['event_electionType'].str.strip()
        
        # Group regional elections
        mask = df['event_electionType'].isin(regional_elections)
        df.loc[mask, 'event_electionType'] = 'Regional Election'
        
        # Group presidential details
        mask = df['event_electionType'].isin(presidential_details)
        df.loc[mask, 'event_electionType'] = 'Presidential Detail'
        
        # Group remaining rare categories
        major_types = ['Presidential', 'Presidential Primary', 'Senate', 'Parliamentary', 
                      'Vice Presidential', 'Prime Minister', 'Regional Election', 
                      'Presidential Detail', 'Balance of Power']
        mask = ~df['event_electionType'].isin(major_types)
        df.loc[mask, 'event_electionType'] = 'Other Election'

# Group rare countries
if 'event_country' in X_train.columns:
    print("\nGrouping rare countries...")
    
    # Define country groupings
    major_countries = ['United States', 'United Kingdom', 'Germany', 'Ireland', 'Brazil', 'Canada']
    eastern_europe = ['Belarus', 'Croatia', 'Romania', 'Moldova', 'Lithuania']
    latin_america = ['Venezuela', 'Mexico', 'Uruguay', 'Argentina', 'El Salvador']
    
    # Apply groupings to both train and test sets
    for df in [X_train, X_test]:
        # Remove leading/trailing spaces
        df['event_country'] = df['event_country'].str.strip()
        
        # Keep major countries
        # Group Eastern Europe
        mask = df['event_country'].isin(eastern_europe)
        df.loc[mask, 'event_country'] = 'Eastern Europe'
        
        # Group Latin America
        mask = df['event_country'].isin(latin_america)
        df.loc[mask, 'event_country'] = 'Latin America'
        
        # Group all other rare countries
        mask = ~df['event_country'].isin(major_countries + ['Eastern Europe', 'Latin America'])
        df.loc[mask, 'event_country'] = 'Other Countries'

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

# 16. Apply preprocessing to get a fully numerical representation
print("Applying preprocessing to transform features...")
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

# Get feature names after one-hot encoding
transformed_feature_names = numeric_features.copy()
if categorical_features and hasattr(preprocessor.named_transformers_['cat'].named_steps['onehot'], 'get_feature_names_out'):
    onehot_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
    categorical_feature_names = onehot_encoder.get_feature_names_out(categorical_features)
    transformed_feature_names = transformed_feature_names + list(categorical_feature_names)

print(f"Number of features after transformation: {len(transformed_feature_names)}")

# 17. Apply SMOTE for classification task if needed
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

# 18. Save the processed data
# Save the cleaned and processed dataframes
complete_df.to_csv(os.path.join(OUTPUT_DIR, 'complete_election_data.csv'), index=False)
modeling_df.to_csv(os.path.join(OUTPUT_DIR, 'cleaned_election_data.csv'), index=False)

# Save the original train/test data (before preprocessing) with indexes
X_train.to_csv(os.path.join(OUTPUT_DIR, 'X_train_original.csv'))
X_test.to_csv(os.path.join(OUTPUT_DIR, 'X_test_original.csv'))

# Save the preprocessed datasets (fully numerical)
np.save(os.path.join(OUTPUT_DIR, 'X_train_preprocessed.npy'), X_train_preprocessed)
np.save(os.path.join(OUTPUT_DIR, 'X_test_preprocessed.npy'), X_test_preprocessed)

# Save the target variables
for target in available_targets:
    pd.DataFrame(y_train_dict[target]).to_csv(os.path.join(OUTPUT_DIR, f'y_train_{target}.csv'))
    pd.DataFrame(y_test_dict[target]).to_csv(os.path.join(OUTPUT_DIR, f'y_test_{target}.csv'))
    np.save(os.path.join(OUTPUT_DIR, f'y_train_{target}.npy'), y_train_dict[target].values)
    np.save(os.path.join(OUTPUT_DIR, f'y_test_{target}.npy'), y_test_dict[target].values)

# Save SMOTE-resampled data if applicable
if 'smote' in X_train_dict:
    np.save(os.path.join(OUTPUT_DIR, 'X_train_smote.npy'), X_train_dict['smote'])
    np.save(os.path.join(OUTPUT_DIR, f'y_train_prediction_correct_smote.npy'), y_train_dict['prediction_correct_smote'])

# Save the feature names for later interpretation
pd.DataFrame({'feature': transformed_feature_names}).to_csv(os.path.join(OUTPUT_DIR, 'transformed_feature_names.csv'), index=False)
pd.DataFrame({'feature': X_train.columns.tolist()}).to_csv(os.path.join(OUTPUT_DIR, 'original_feature_names.csv'), index=False)

# Save the preprocessor for future use
joblib.dump(preprocessor, os.path.join(OUTPUT_DIR, 'feature_preprocessor.joblib'))

print("\nData preparation complete. Files saved to the modified_analysis directory.")