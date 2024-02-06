import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.dates import DateFormatter
path = '../data/hos_data/'

# Constants
PATH = '../data/hos_data/'
FILTERED_DATA_FILE = 'filtered_data.csv'
COVID19_DATA_FILE = 'covid19_data_from_april_8.csv'
DATE_COLUMN = 'date'
REGION_MAPPING = {
    'North East England': 'North East and Yorkshire',
    'Yorkshire and the Humber': 'North East and Yorkshire',
    'East Midlands': 'Midlands',
    'West Midlands': 'Midlands',
    'East of England': 'East of England',
    'London Region': 'London',
    'South East England': 'South East',
    'South West England': 'South West',
    'North West England': 'North West'
}

def load_and_prepare_data(path, filename, date_column):
    """Load data from CSV file and prepare it."""
    df = pd.read_csv(f"{path}{filename}")
    df[date_column] = pd.to_datetime(df[date_column])
    return df

def map_regions_and_merge(df1, df2, mapping, merge_columns, drop_columns):
    """Map regions in one dataframe to another and merge them."""
    df2['mapped_region'] = df2['region'].map(mapping)
    merged_df = pd.merge(df1, df2, how='inner', left_on=merge_columns, right_on=['date', 'mapped_region'])
    merged_df.drop_duplicates(inplace=True)
    merged_df.drop(columns=drop_columns, inplace=True, errors='ignore')
    return merged_df
# Load and prepare data
filtered_data = load_and_prepare_data(PATH, FILTERED_DATA_FILE, DATE_COLUMN)
covid19_data = load_and_prepare_data(PATH, COVID19_DATA_FILE, DATE_COLUMN)

# Merge dataframes
merge_columns = ['date', 'areaName']
drop_columns = ['Unnamed: 0', 'mapped_region', 'areaType', 'areaCode', 'region']
merged_data = map_regions_and_merge(filtered_data, covid19_data, REGION_MAPPING, merge_columns, drop_columns)

# 1. Time Series Graph of New Confirmed Cases Over Time for Each NHS Region
plt.figure(figsize=(14, 8))
for region in merged_data['areaName'].unique():
    region_data = merged_data[merged_data['areaName'] == region]
    plt.plot(region_data['date'], region_data['new_confirmed'], label=region)

plt.title('New Confirmed COVID-19 Cases Over Time by NHS Region')
plt.xlabel('Date')
plt.ylabel('New Confirmed Cases')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 2. Bar Chart Comparing Cumulative Confirmed Cases and Deaths Across NHS Regions
cumulative_data = merged_data.groupby('areaName').agg({
    'cumulative_confirmed': 'max',
    'cumulative_deceased': 'max'
}).reset_index()

plt.figure(figsize=(14, 8))
x = range(len(cumulative_data))
plt.bar(x, cumulative_data['cumulative_confirmed'], width=0.4, label='Cumulative Confirmed Cases', align='center')
plt.bar(x, cumulative_data['cumulative_deceased'], width=0.4, label='Cumulative Deaths', align='edge')
plt.xlabel('NHS Region')
plt.ylabel('Counts')
plt.title('Cumulative Confirmed Cases and Deaths by NHS Region')
plt.xticks(x, cumulative_data['areaName'], rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# 3. Scatter Plot: New Confirmed Cases vs. New Admissions
plt.figure(figsize=(10, 6))
sns.scatterplot(data=merged_data, x='new_confirmed', y='newAdmissions', hue='areaName', style='areaName')
plt.title('New Confirmed Cases vs. New Admissions by NHS Region')
plt.xlabel('New Confirmed Cases')
plt.ylabel('New Admissions')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# 4. Heat Map of New Confirmed Cases Over Time by Region
# Pivot data for heat map
pivot_cases_time_region = merged_data.pivot_table(index='date', columns='areaName', values='new_confirmed', aggfunc='sum').fillna(0)

plt.figure(figsize=(14, 10))
sns.heatmap(pivot_cases_time_region, cmap='viridis')
plt.title('Heat Map of New Confirmed COVID-19 Cases Over Time by NHS Region')
plt.xlabel('NHS Region')
plt.ylabel('Date')
plt.xticks(rotation=45)
plt.show()

# 5. Box Plot for Hospital Cases by Region
plt.figure(figsize=(14, 8))
sns.boxplot(x='areaName', y='hospitalCases', data=merged_data)
plt.title('Distribution of Hospital Cases by NHS Region')
plt.xlabel('NHS Region')
plt.ylabel('Hospital Cases')
plt.xticks(rotation=45)
plt.show()

# 6. Correlation Heat Map
# Select relevant metrics for correlation
metrics_for_correlation = merged_data[['new_confirmed', 'new_deceased', 'hospitalCases', 'newAdmissions', 'cumulative_confirmed', 'cumulative_deceased']]
correlation_matrix = metrics_for_correlation.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heat Map of COVID-19 Metrics')
plt.show()



