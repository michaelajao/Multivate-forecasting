import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.dates import DateFormatter
path = '../data/hos_data/'

filtered_data = pd.read_csv(path + 'filtered_data.csv')
covid19_data = pd.read_csv(path + 'covid19_data_from_april_8.csv')

filtered_data['date'] = pd.to_datetime(filtered_data['date'])
covid19_data['date'] = pd.to_datetime(covid19_data['date'])

filtered_data_names = filtered_data['areaName'].unique()
covid19_names = covid19_data['region'].unique()
# Map regions from covid19_data to match those in filtered_data
region_mapping = {
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

# Apply mapping to covid19_data 'region' column
covid19_data['mapped_region'] = covid19_data['region'].map(region_mapping)


# 1. Time Series Graph of New Confirmed Cases Over Time for Each NHS Region
plt.figure(figsize=(14, 8))
for region in merged_data_corrected['areaName'].unique():
    region_data = merged_data_corrected[merged_data_corrected['areaName'] == region]
    plt.plot(region_data['date'], region_data['new_confirmed'], label=region)

plt.title('New Confirmed COVID-19 Cases Over Time by NHS Region')
plt.xlabel('Date')
plt.ylabel('New Confirmed Cases')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 2. Bar Chart Comparing Cumulative Confirmed Cases and Deaths Across NHS Regions
cumulative_data = merged_data_corrected.groupby('areaName').agg({
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
sns.scatterplot(data=merged_data_corrected, x='new_confirmed', y='newAdmissions', hue='areaName', style='areaName')
plt.title('New Confirmed Cases vs. New Admissions by NHS Region')
plt.xlabel('New Confirmed Cases')
plt.ylabel('New Admissions')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()



# 4. Heat Map of New Confirmed Cases Over Time by Region
# Pivot data for heat map
pivot_cases_time_region = merged_data_corrected.pivot_table(index='date', columns='areaName', values='new_confirmed', aggfunc='sum').fillna(0)

plt.figure(figsize=(14, 10))
sns.heatmap(pivot_cases_time_region, cmap='viridis')
plt.title('Heat Map of New Confirmed COVID-19 Cases Over Time by NHS Region')
plt.xlabel('NHS Region')
plt.ylabel('Date')
plt.xticks(rotation=45)
plt.show()

# 5. Box Plot for Hospital Cases by Region
plt.figure(figsize=(14, 8))
sns.boxplot(x='areaName', y='hospitalCases', data=merged_data_corrected)
plt.title('Distribution of Hospital Cases by NHS Region')
plt.xlabel('NHS Region')
plt.ylabel('Hospital Cases')
plt.xticks(rotation=45)
plt.show()

# 6. Correlation Heat Map
# Select relevant metrics for correlation
metrics_for_correlation = merged_data_corrected[['new_confirmed', 'new_deceased', 'hospitalCases', 'newAdmissions', 'cumulative_confirmed', 'cumulative_deceased']]
correlation_matrix = metrics_for_correlation.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heat Map of COVID-19 Metrics')
plt.show()


# Perform the merge again using the new 'mapped_region' for alignment with 'areaName'
merged_data_corrected = pd.merge(filtered_data, covid19_data, how='inner', left_on=['date', 'areaName'], right_on=['date', 'mapped_region'])
merged_data_cleaned = merged_data_corrected.drop_duplicates()
columns_to_drop = ['Unnamed: 0', 'mapped_region', 'areaType', 'areaCode', 'region', 'cumulative_confirmed', 'cumulative_deceased']
merged_data_cleaned.drop(columns=columns_to_drop, inplace=True, errors='ignore')
