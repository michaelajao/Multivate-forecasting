import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib as mpl


mpl.rcParams['figure.figsize'] = (10, 6)
mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.titlesize'] = 16
# mpl.rcParams['axes.labelsize'] = 14
# mpl.rcParams['xtick.labelsize'] = 12
# mpl.rcParams['ytick.labelsize'] = 12
# mpl.rcParams['legend.fontsize'] = 12



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
# plt.figure(figsize=(14, 8))
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

# plt.figure(figsize=(15, 10))
for region in merged_data['areaName'].unique():
    region_data = merged_data[merged_data['areaName'] == region]
    plt.plot(region_data['date'], region_data['cumulative_confirmed'], label=region)
    
plt.title('Cumulative Confirmed COVID-19 Cases Over Time by NHS Region')
plt.xlabel('Date')
plt.ylabel('Cumulative Confirmed Cases')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# data_image = merged_data[(merged_data['date'] >= '2020-01-01') & (merged_data['date'] <= '2021-12-31')]

# # Prepare the figure and subplots with improved readability
# fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
# fig.subplots_adjust(hspace=0.5)

# # Define plot details for new and cumulative confirmed cases
# metrics = ['new_confirmed', 'cumulative_confirmed']
# colors = ['orange', 'red']
# titles = ['Daily New Confirmed COVID-19 Cases Over Time', 'Cumulative Confirmed COVID-19 Cases Over Time']

# # Assuming 'data_image' already filtered as per your dataset
# for ax, metric, color, title in zip(axs, metrics, colors, titles):
#     # For each NHS region, plot the data
#     for region in data_image['areaName'].unique():
#         region_data = data_image[data_image['areaName'] == region]
#         ax.plot(region_data['date'], region_data[metric], label=region, color=color, alpha=0.75)
#     ax.set_title(title, fontsize=14)
#     ax.set_ylabel('Count', fontsize=12)
#     ax.tick_params(axis='x', labelrotation=45)
#     ax.tick_params(axis='both', labelsize=10)
#     ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
#     ax.xaxis.set_major_locator(mdates.MonthLocator())

# # Add a legend outside the last plot
# handles, labels = axs[0].get_legend_handles_labels()
# fig.legend(handles, labels, loc='upper center', ncol=3, fontsize='small')

# # Set common x-label
# axs[-1].set_xlabel('Date', fontsize=12)

# # Improve overall layout and save to PDF
# plt.tight_layout()

# pdf_path_improved = '/mnt/data/trend_analysis_improved.pdf'  # Adjust path as needed
# # with PdfPages(pdf_path_improved) as pdf:
# #     pdf.savefig(fig, bbox_inches='tight')

# # pdf_path_improved

# # 2. Bar Chart Comparing Cumulative Confirmed Cases and Deaths Across NHS Regions
# cumulative_data = merged_data.groupby('areaName').agg({
#     'cumulative_confirmed': 'max',
#     'cumulative_deceased': 'max'
# }).reset_index()

# plt.figure(figsize=(14, 8))
# x = range(len(cumulative_data))
# plt.bar(x, cumulative_data['cumulative_confirmed'], width=0.4, label='Cumulative Confirmed Cases', align='center')
# plt.bar(x, cumulative_data['cumulative_deceased'], width=0.4, label='Cumulative Deaths', align='edge')
# plt.xlabel('NHS Region')
# plt.ylabel('Counts')
# plt.title('Cumulative Confirmed Cases and Deaths by NHS Region')
# plt.xticks(x, cumulative_data['areaName'], rotation=45)
# plt.legend()
# plt.tight_layout()
# plt.show()



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



