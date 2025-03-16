import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import os

# Create output directory
output_dir = "feature_relationships"
os.makedirs(output_dir, exist_ok=True)

# Load your cleaned data
df = pd.read_csv("modified_analysis/cleaned_election_data.csv")

# Select top features identified in your importance analysis
top_features = [
    'price_range', 'unique_traders_count', 'price_fluctuations', 
    'volumeNum', 'final_week_momentum', 'buy_sell_ratio',
    'event_commentCount', 'price_volatility', 'volume_acceleration', 
    'trader_concentration'
]

# Add the target variable
analysis_features = top_features + ['brier_score']

# Create a correlation matrix
correlation_df = df[analysis_features].corr()

# Visualize correlation matrix
plt.figure(figsize=(12, 10))
mask = np.triu(np.ones_like(correlation_df, dtype=bool))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(correlation_df, mask=mask, cmap=cmap, vmax=.5, vmin=-.5, 
            square=True, linewidths=.5, annot=True, fmt='.2f')
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig(f"{output_dir}/feature_correlation_matrix.png", dpi=300)
plt.close()

# Create pairplots for the most important features (top 5)
sns.pairplot(df[top_features[:5] + ['brier_score']], 
             diag_kind='kde', 
             plot_kws={'alpha': 0.6, 's': 30, 'edgecolor': 'k'})
plt.savefig(f"{output_dir}/top_features_pairplot.png", dpi=300)
plt.close()

# Analyze feature interactions with brier_score
for feature in top_features:
    plt.figure(figsize=(10, 6))
    
    # Create scatter plot with trend line
    sns.regplot(x=feature, y='brier_score', data=df, scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
    
    # Calculate correlation
    corr = df[[feature, 'brier_score']].corr().iloc[0,1]
    
    plt.title(f'Relationship between {feature} and Brier Score (corr={corr:.3f})')
    plt.savefig(f"{output_dir}/{feature}_vs_brier.png", dpi=300)
    plt.close()

# Create a DataFrame with feature correlations to brier_score
feature_corrs = pd.DataFrame({
    'feature': top_features,
    'correlation_with_brier': [correlation_df.loc[feature, 'brier_score'] for feature in top_features]
}).sort_values('correlation_with_brier', key=abs, ascending=False)

feature_corrs.to_csv(f"{output_dir}/feature_correlations_with_brier.csv", index=False)

print("Feature relationship analysis complete!")