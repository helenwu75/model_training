import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data with low_memory=False to avoid the DtypeWarning
df = pd.read_csv('final_election_results.csv', low_memory=False)

# Extract the columns of interest
election_types = df['event_electionType'].value_counts()
regions = df['event_country'].value_counts()

# Create election type distribution table
election_type_table = pd.DataFrame({
    'Election Type': election_types.index,
    'Count': election_types.values,
    'Percentage': (election_types.values / len(df) * 100).round(1)
})

# Create region distribution table
region_table = pd.DataFrame({
    'Region': regions.index,
    'Count': regions.values,
    'Percentage': (regions.values / len(df) * 100).round(1)
})

# Save tables to CSV for easy inclusion in paper
election_type_table.to_csv('election_type_distribution.csv', index=False)
region_table.to_csv('region_distribution.csv', index=False)

# Set up the visualization style - using updated style syntax
sns.set()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot election types (limiting to top 8 for readability)
top_types = election_types.iloc[:8]
other_types_count = election_types.iloc[8:].sum()
if len(election_types) > 8:
    # Create a new Series with 'Other' added
    top_types_with_other = pd.Series(top_types.values, index=top_types.index)
    top_types_with_other['Other'] = other_types_count
    top_types = top_types_with_other

sns.barplot(x=top_types.values, y=top_types.index, ax=ax1)
ax1.set_title('Distribution by Election Type')
ax1.set_xlabel('Number of Markets')

# Plot regions (limiting to top 8 for readability)
top_regions = regions.iloc[:8]
other_regions_count = regions.iloc[8:].sum()
if len(regions) > 8:
    # Create a new Series with 'Other' added
    top_regions_with_other = pd.Series(top_regions.values, index=top_regions.index)
    top_regions_with_other['Other'] = other_regions_count
    top_regions = top_regions_with_other

sns.barplot(x=top_regions.values, y=top_regions.index, ax=ax2)
ax2.set_title('Distribution by Region')
ax2.set_xlabel('Number of Markets')

plt.tight_layout()
plt.savefig('market_distribution.png', dpi=300)
print("Analysis complete. Check the current directory for the output files.")