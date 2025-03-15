"""
Generate Descriptive Statistics for Election Market Data

This script generates comprehensive descriptive statistics for all numerical features
in the cleaned election market dataset, including:
- Basic statistics (mean, median, min, max, std)
- Distribution characteristics (skewness, kurtosis)
- Percentiles
- Missing value information

Output is saved as a CSV file with a row for each feature.
"""

import pandas as pd
import numpy as np
import os
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Path configuration
INPUT_DIR = "modified_analysis"
OUTPUT_DIR = "descriptive_stats"
INPUT_FILE = "cleaned_election_data.csv"
OUTPUT_FILE = "feature_descriptive_stats.csv"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def calculate_descriptive_stats(df):
    """
    Calculate descriptive statistics for numerical features
    
    Args:
        df: Pandas DataFrame with features
        
    Returns:
        DataFrame with descriptive statistics
    """
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=['number'])
    
    # Dictionary to store results
    stats_dict = {}
    
    # Calculate statistics for each numeric column
    for col in numeric_df.columns:
        # Get column data, dropping NaN values
        data = numeric_df[col].dropna()
        
        # Skip if no valid data
        if len(data) == 0:
            continue
            
        # Basic statistics
        stats_dict[col] = {
            'count': len(data),
            'missing_count': df[col].isna().sum(),
            'missing_pct': (df[col].isna().sum() / len(df) * 100),
            'mean': data.mean(),
            'median': data.median(),
            'std': data.std(),
            'min': data.min(),
            'max': data.max(),
            'range': data.max() - data.min(),
            
            # Percentiles
            'p5': np.percentile(data, 5),
            'p25': np.percentile(data, 25),
            'p75': np.percentile(data, 75),
            'p95': np.percentile(data, 95),
            'iqr': np.percentile(data, 75) - np.percentile(data, 25),
            
            # Distribution characteristics
            'skewness': stats.skew(data),
            'kurtosis': stats.kurtosis(data)
        }
        
        # Calculate coefficient of variation (CV)
        if data.mean() != 0:
            stats_dict[col]['cv'] = (data.std() / abs(data.mean())) * 100
        else:
            stats_dict[col]['cv'] = np.nan
    
    # Convert to DataFrame
    stats_df = pd.DataFrame.from_dict(stats_dict, orient='index')
    
    # Rearrange columns to a more logical order
    column_order = [
        'count', 'missing_count', 'missing_pct',
        'mean', 'median', 'std', 'cv',
        'min', 'p5', 'p25', 'p75', 'p95', 'max', 'range', 'iqr',
        'skewness', 'kurtosis'
    ]
    column_order = [col for col in column_order if col in stats_df.columns]
    stats_df = stats_df[column_order]
    
    # Add feature name as a column
    stats_df = stats_df.reset_index().rename(columns={'index': 'feature'})
    
    return stats_df

def generate_distribution_plots(df, output_dir):
    """
    Generate distribution plots for numerical features
    
    Args:
        df: Pandas DataFrame with features
        output_dir: Directory to save plots
    """
    plots_dir = os.path.join(output_dir, "distribution_plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    numeric_df = df.select_dtypes(include=['number'])
    
    # Generate plots for each feature
    for col in numeric_df.columns:
        plt.figure(figsize=(10, 6))
        
        # Create subplot grid
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Histogram with KDE
        sns.histplot(numeric_df[col].dropna(), kde=True, ax=ax1)
        ax1.set_title(f'Distribution of {col}')
        ax1.axvline(numeric_df[col].mean(), color='red', linestyle='--', 
                   label=f'Mean: {numeric_df[col].mean():.2f}')
        ax1.axvline(numeric_df[col].median(), color='green', linestyle='-.', 
                   label=f'Median: {numeric_df[col].median():.2f}')
        ax1.legend()
        
        # Box plot
        sns.boxplot(x=numeric_df[col].dropna(), ax=ax2)
        ax2.set_title(f'Boxplot of {col}')
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"{col}_distribution.png"))
        plt.close()
        
    print(f"Distribution plots saved to {plots_dir}")

def main():
    # Load the dataset
    data_path = os.path.join(INPUT_DIR, INPUT_FILE)
    print(f"Loading data from {data_path}")
    
    try:
        df = pd.read_csv(data_path)
        print(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")
    except FileNotFoundError:
        print(f"Error: File {data_path} not found.")
        print("Checking for alternative input files...")
        
        # Attempt to locate an alternative file
        if os.path.exists(INPUT_DIR):
            files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.csv')]
            if files:
                print(f"Found alternative files: {files}")
                print(f"Using {files[0]}")
                df = pd.read_csv(os.path.join(INPUT_DIR, files[0]))
                print(f"Loaded alternative dataset with {df.shape[0]} rows and {df.shape[1]} columns")
            else:
                print(f"No CSV files found in {INPUT_DIR}")
                return
        else:
            print(f"Directory {INPUT_DIR} not found")
            
            # Try loading the file directly from the current directory
            if os.path.exists("mini_election_metrics_results_edit.csv"):
                print("Found mini_election_metrics_results_edit.csv in current directory")
                df = pd.read_csv("mini_election_metrics_results_edit.csv")
                print(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")
            else:
                print("Could not find any suitable input files")
                return
    
    # Calculate descriptive statistics
    print("Calculating descriptive statistics...")
    stats_df = calculate_descriptive_stats(df)
    
    # Save statistics to CSV
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    stats_df.to_csv(output_path, index=False)
    print(f"Descriptive statistics saved to {output_path}")
    
    # Generate distribution plots
    print("Generating distribution plots...")
    generate_distribution_plots(df, OUTPUT_DIR)
    
    # Generate summary Markdown report
    md_output = os.path.join(OUTPUT_DIR, "descriptive_stats_report.md")
    
    # Function to convert dataframe to markdown table
    def df_to_markdown(df):
        # Format the headers
        markdown_table = "| " + " | ".join(df.columns) + " |\n"
        # Add the separator line
        markdown_table += "| " + " | ".join(["---" for _ in df.columns]) + " |\n"
        
        # Format the data, handling floats appropriately
        for _, row in df.iterrows():
            formatted_row = []
            for val in row:
                if isinstance(val, float):
                    formatted_row.append(f"{val:.4f}")
                else:
                    formatted_row.append(str(val))
            markdown_table += "| " + " | ".join(formatted_row) + " |\n"
        
        return markdown_table
    
    # Create statistics table in markdown format
    # For better readability, split into multiple tables by statistic groups
    
    # First group: Basic counts and presence
    count_cols = ['feature', 'count', 'missing_count', 'missing_pct']
    count_df = stats_df[count_cols].copy()
    count_md = df_to_markdown(count_df)
    
    # Second group: Central tendency and dispersion
    central_cols = ['feature', 'mean', 'median', 'std', 'cv']
    central_df = stats_df[central_cols].copy()
    central_md = df_to_markdown(central_df)
    
    # Third group: Range and percentiles
    range_cols = ['feature', 'min', 'p5', 'p25', 'p75', 'p95', 'max', 'range', 'iqr']
    range_df = stats_df[range_cols].copy()
    range_md = df_to_markdown(range_df)
    
    # Fourth group: Distribution shape
    shape_cols = ['feature', 'skewness', 'kurtosis']
    shape_df = stats_df[shape_cols].copy()
    shape_md = df_to_markdown(shape_df)
    
    # Create Markdown report
    md_content = f"""# Descriptive Statistics - Election Market Data

*Generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*

## Dataset Overview

- **Dataset dimensions**: {df.shape[0]} rows × {df.shape[1]} columns
- **Numeric features**: {len(df.select_dtypes(include=['number']).columns)}

## Feature Statistics

### Data Completeness

{count_md}

### Central Tendency and Dispersion

{central_md}

### Range and Percentiles

{range_md}

### Distribution Shape

{shape_md}

## Interpretation Guide

- **count**: Number of non-missing observations
- **missing_count**: Number of missing values
- **missing_pct**: Percentage of missing values
- **mean**: Average value
- **median**: Middle value (50th percentile)
- **std**: Standard deviation
- **cv**: Coefficient of variation (std/mean × 100)
- **min**: Minimum value
- **max**: Maximum value
- **range**: Difference between max and min
- **p5, p25, p75, p95**: 5th, 25th, 75th, and 95th percentiles
- **iqr**: Interquartile range (p75-p25)
- **skewness**: Measure of asymmetry (0 = symmetric, >0 = right-skewed, <0 = left-skewed)
- **kurtosis**: Measure of "tailedness" (0 = normal, >0 = heavier tails, <0 = lighter tails)

## Distribution Plots

Distribution plots for each feature are available in the `distribution_plots` directory.
"""
    
    # Save Markdown report
    with open(md_output, 'w') as f:
        f.write(md_content)
    
    print(f"Markdown report saved to {md_output}")
    print("Analysis complete!")

if __name__ == "__main__":
    main()