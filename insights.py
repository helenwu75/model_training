import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.tree import export_text

# Create output directory
output_dir = "actionable_insights"
os.makedirs(output_dir, exist_ok=True)

# Load your cleaned data
df = pd.read_csv("modified_analysis/cleaned_election_data.csv")

# Select top features for our analysis
top_features = [
    'price_range', 'unique_traders_count', 'price_fluctuations', 
    'volumeNum', 'final_week_momentum', 'buy_sell_ratio',
    'price_volatility', 'volume_acceleration', 'trader_concentration'
]

# Create a categorization of market quality (good/poor prediction)
median_brier = df['brier_score'].median()
df['market_quality'] = np.where(df['brier_score'] <= median_brier, 'Good', 'Poor')

# Simple decision tree to find actionable thresholds
X = df[top_features]
y = (df['brier_score'] <= median_brier).astype(int)  # Binary target: 1 for good, 0 for poor

# Fit a simple decision tree with limited depth for interpretability
dt = DecisionTreeRegressor(max_depth=3, min_samples_leaf=20)
dt.fit(X, y)

# Export the decision tree rules
tree_rules = export_text(dt, feature_names=top_features)
with open(f"{output_dir}/decision_tree_rules.txt", "w") as f:
    f.write(tree_rules)

# Visualize the decision tree
plt.figure(figsize=(15, 10))
plot_tree(dt, filled=True, feature_names=top_features, 
          class_names=['Poor Quality', 'Good Quality'],
          rounded=True, proportion=True, precision=2)
plt.savefig(f"{output_dir}/decision_tree_visualization.png", dpi=300)
plt.close()

# Create key threshold table based on feature distributions by market quality
thresholds = []

for feature in top_features:
    # Calculate median for good markets
    good_median = df[df['market_quality'] == 'Good'][feature].median()
    # Calculate median for poor markets
    poor_median = df[df['market_quality'] == 'Poor'][feature].median()
    
    # Determine direction (higher or lower is better)
    direction = "Lower is better" if good_median < poor_median else "Higher is better"
    
    # Calculate suggested threshold (25th or 75th percentile of good markets)
    if direction == "Lower is better":
        threshold = df[df['market_quality'] == 'Good'][feature].quantile(0.75)
    else:
        threshold = df[df['market_quality'] == 'Good'][feature].quantile(0.25)
    
    thresholds.append({
        'feature': feature,
        'good_markets_median': good_median,
        'poor_markets_median': poor_median,
        'direction': direction,
        'suggested_threshold': threshold
    })

thresholds_df = pd.DataFrame(thresholds)
thresholds_df.to_csv(f"{output_dir}/market_quality_thresholds.csv", index=False)

# Create practical indicators scale
# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[top_features])
X_scaled_df = pd.DataFrame(X_scaled, columns=top_features)

# Align directions so higher is always better
for i, feature in enumerate(top_features):
    row = thresholds_df[thresholds_df['feature'] == feature].iloc[0]
    if row['direction'] == "Lower is better":
        X_scaled_df[feature] = -X_scaled_df[feature]

# Create a composite score (average of aligned standardized features)
df['quality_score'] = X_scaled_df[top_features].mean(axis=1)

# Visualize the relationship between quality score and brier score
plt.figure(figsize=(10, 6))
sns.scatterplot(x='quality_score', y='brier_score', data=df)
sns.regplot(x='quality_score', y='brier_score', data=df, scatter=False, color='red')
plt.title('Market Quality Score vs Brier Score')
plt.xlabel('Quality Score (higher is better)')
plt.ylabel('Brier Score (lower is better)')
plt.savefig(f"{output_dir}/market_quality_score_vs_brier.png", dpi=300)
plt.close()

# Define quality score ranges and labels
df['quality_category'] = pd.qcut(df['quality_score'], 
                             q=[0, 0.25, 0.5, 0.75, 1.0],
                             labels=['Very Poor', 'Poor', 'Good', 'Excellent'])

# Calculate brier score statistics by quality category
quality_stats = df.groupby('quality_category')['brier_score'].agg(['mean', 'median', 'std', 'count']).reset_index()
quality_stats.to_csv(f"{output_dir}/quality_category_stats.csv", index=False)

# Create a simple scoring system
scoring_rules = []
for feature in top_features:
    row = thresholds_df[thresholds_df['feature'] == feature].iloc[0]
    
    if row['direction'] == "Lower is better":
        rule = f"If {feature} <= {row['suggested_threshold']:.4f}, assign +1 point"
    else:
        rule = f"If {feature} >= {row['suggested_threshold']:.4f}, assign +1 point"
    
    scoring_rules.append(rule)

with open(f"{output_dir}/market_quality_scoring_system.txt", "w") as f:
    f.write("POLYMARKET PREDICTION QUALITY SCORING SYSTEM\n")
    f.write("-------------------------------------------\n\n")
    f.write("Instructions: Score each market based on the following criteria.\n")
    f.write("Sum the points (maximum score: " + str(len(scoring_rules)) + ").\n\n")
    
    for i, rule in enumerate(scoring_rules, 1):
        f.write(f"{i}. {rule}\n")
    
    f.write("\nINTERPRETING THE SCORE:\n")
    f.write("0-2 points: Very poor prediction quality\n")
    f.write(f"3-{len(scoring_rules)//2} points: Moderate prediction quality\n")
    f.write(f"{len(scoring_rules)//2 + 1}-{len(scoring_rules)-1} points: Good prediction quality\n")
    f.write(f"{len(scoring_rules)} points: Excellent prediction quality\n")

print("Actionable insights development complete!")