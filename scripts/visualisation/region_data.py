import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set the default style for plots
plt.style.use("seaborn-v0_8-poster")
plt.rcParams.update({
    "font.size": 20,
    "figure.figsize": [10, 5],
    "figure.facecolor": "white",
    "figure.autolayout": True,
    "figure.dpi": 600,
    "savefig.dpi": 600,
    "savefig.format": "pdf",
    "savefig.bbox": "tight",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "axes.labelsize": 14,
    "axes.titlesize": 18,
    "axes.facecolor": "white",
    "axes.grid": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.formatter.limits": (0, 5),
    "axes.formatter.use_mathtext": True,
    "axes.formatter.useoffset": False,
    "axes.xmargin": 0,
    "axes.ymargin": 0,
    "legend.fontsize": 14,
    "legend.frameon": False,
    "legend.loc": "best",
    "lines.linewidth": 2,
    "lines.markersize": 8,
    "xtick.labelsize": 14,
    "xtick.direction": "in",
    "xtick.top": False,
    "ytick.labelsize": 14,
    "ytick.direction": "in",
    "ytick.right": False,
    "grid.color": "grey",
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
    "errorbar.capsize": 4,
    "figure.subplot.wspace": 0.4,
    "figure.subplot.hspace": 0.4,
    "image.cmap": "viridis",
})

# Constants
data_path = "../../data/hos_data/"
filtered_data_file = "filtered_data.csv"
covid19_data_file = "covid19_data_from_april_8.csv"
date_column = "date"

# Load the initial data
filtered_data = pd.read_csv(data_path + filtered_data_file)
covid19_data = pd.read_csv(data_path + covid19_data_file).drop(columns=['Unnamed: 0',])

# Convert date columns to datetime
filtered_data['date'] = pd.to_datetime(filtered_data['date'])
covid19_data['date'] = pd.to_datetime(covid19_data['date'])

# Apply region mapping to both datasets
region_mapping = {
    "North East England": "North East and Yorkshire",
    "Yorkshire and the Humber": "North East and Yorkshire",
    "East Midlands": "Midlands",
    "West Midlands": "Midlands",
    "East of England": "East of England",
    "London Region": "London",
    "South East England": "South East",
    "South West England": "South West",
    "North West England": "North West",
}

# Correctly map the regions
filtered_data['region'] = filtered_data['areaName'].replace(region_mapping)
covid19_data['region'] = covid19_data['region'].replace(region_mapping)

# Handle NaN values in the region column
filtered_data = filtered_data.dropna(subset=['region'])
covid19_data = covid19_data.dropna(subset=['region'])

# Filter for common dates
common_dates = pd.to_datetime(list(set(filtered_data['date']).intersection(set(covid19_data['date']))))

# check if the common dates are continuous
date_range = pd.date_range(start=common_dates.min(), end=common_dates.max())
missing_dates = date_range.difference(common_dates)
if not missing_dates.empty:
    print(f"Missing dates: {missing_dates}")
    
# Filter the data for common dates
filtered_data = filtered_data[filtered_data['date'].isin(common_dates)]
covid19_data = covid19_data[covid19_data['date'].isin(common_dates)]

# Identify missing dates for each region
def identify_missing_dates(df, date_col, region_col):
    missing_dates_info = {}
    regions = df[region_col].unique()
    for region in regions:
        region_data = df[df[region_col] == region]
        date_range = pd.date_range(start=region_data[date_col].min(), end=region_data[date_col].max())
        missing_dates = date_range.difference(region_data[date_col])
        if not missing_dates.empty:
            missing_dates_info[region] = missing_dates
    return missing_dates_info

missing_dates_info_filtered = identify_missing_dates(filtered_data, 'date', 'region')
missing_dates_info_covid19 = identify_missing_dates(covid19_data, 'date', 'region')

# Fill missing dates for each region
def fill_missing_dates(df, date_col, region_col, missing_dates_info):
    complete_data = []
    for region, missing_dates in missing_dates_info.items():
        region_data = df[df[region_col] == region]
        for date in missing_dates:
            missing_row = {date_col: date, region_col: region}
            for col in df.columns:
                if col not in [date_col, region_col]:
                    missing_row[col] = 0  # or np.nan if preferred
            complete_data.append(missing_row)
    complete_data_df = pd.DataFrame(complete_data)
    df = pd.concat([df, complete_data_df]).sort_values(by=[region_col, date_col]).reset_index(drop=True)
    return df

filtered_data_complete = fill_missing_dates(filtered_data, 'date', 'region', missing_dates_info_filtered)
covid19_data_complete = fill_missing_dates(covid19_data, 'date', 'region', missing_dates_info_covid19)

# Verify date continuity after filling missing dates
def check_date_continuity(df, date_col, region_col):
    regions = df[region_col].unique()
    for region in regions:
        region_data = df[df[region_col] == region]
        date_range = pd.date_range(start=region_data[date_col].min(), end=region_data[date_col].max())
        missing_dates = date_range.difference(region_data[date_col])
        if not missing_dates.empty:
            print(f"Missing dates for {region}: {missing_dates}")

check_date_continuity(filtered_data_complete, 'date', 'region')
check_date_continuity(covid19_data_complete, 'date', 'region')

# Group by date and new region, then sum the values for filtered_data
filtered_data_grouped = filtered_data_complete.groupby(['date', 'region']).sum(numeric_only=True).reset_index()

# Group by date and new region, then sum the values for covid19_data
covid19_data_grouped = covid19_data_complete.groupby(['date', 'region']).sum(numeric_only=True).reset_index()

# Merge the datasets on date and region
merged_data = pd.merge(filtered_data_grouped, covid19_data_grouped, on=['date', 'region'], how='inner')

# Verify the merged data includes all expected regions
unique_regions_merged_data = merged_data['region'].unique()
print(f"Unique regions in merged_data: {unique_regions_merged_data}")

# Save the merged data as CSV and pickle
merged_data.to_csv(os.path.join(data_path, "merged_data.csv"), index=False)
merged_data.to_pickle(os.path.join(data_path, "merged_data.pkl"))

merged_data["date"] = pd.to_datetime(merged_data["date"])

# Create a new dataset called "England data" by aggregating the data from all regions
england_data = merged_data.groupby('date').sum(numeric_only=True).reset_index()

# Add a column to indicate the region as "England"
england_data['region'] = 'England'

# Save the England data as CSV and pickle
england_data.to_csv(os.path.join(data_path, "england_data.csv"), index=False)
england_data.to_pickle(os.path.join(data_path, "england_data.pkl"))

england_data["date"] = pd.to_datetime(england_data["date"])

# Trend Analysis for England
plt.figure(figsize=(14, 8))
plt.plot(england_data['date'], england_data['new_confirmed'], label='England')
plt.xlabel('Date')
plt.ylabel('New Confirmed Cases')
plt.title('Trend of New Confirmed Cases Over Time in England')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('../../reports/images/trend_new_confirmed_cases_england.pdf')
plt.show()

# Detailed Summary Statistics
summary_stats_england = england_data[['new_confirmed', 'new_deceased', 'hospitalCases']].describe()
print(summary_stats_england)

# Histogram of New Confirmed Cases for England
plt.figure(figsize=(12, 6))
plt.hist(england_data['new_confirmed'], bins=50, color='blue', alpha=0.7)
plt.xlabel('New Confirmed Cases')
plt.ylabel('Frequency')
plt.title('Distribution of New Confirmed Cases in England')
plt.grid(True)
plt.tight_layout()
plt.savefig('../../reports/images/histogram_new_confirmed_cases_england.pdf')
plt.show()

# Box Plot of New Deceased Cases for England
plt.figure(figsize=(12, 6))
sns.boxplot(y='new_deceased', data=england_data)
plt.ylabel('New Deceased Cases')
plt.title('Distribution of New Deceased Cases in England')
plt.grid(True)
plt.tight_layout()
plt.savefig('../../reports/images/boxplot_new_deceased_cases_england.pdf')
plt.show()

# Heatmap of Correlation Matrix for England
plt.figure(figsize=(12, 6))
corr_matrix_england = england_data[['new_confirmed', 'new_deceased', 'hospitalCases']].corr()
sns.heatmap(corr_matrix_england, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix for Key Metrics in England')
plt.tight_layout()
plt.savefig('../../reports/images/heatmap_correlation_matrix_england.pdf')
plt.show()

# Saving merged data


# Convert date to datetime if not already done
merged_data["date"] = pd.to_datetime(merged_data["date"])

# 1. Time Series Graph of New Confirmed Cases Over Time for Each NHS Region
plt.figure(figsize=(12, 6))
for region in merged_data["region"].unique():
    region_data = merged_data[merged_data["region"] == region]
    plt.plot(region_data["date"], region_data["new_confirmed"], label=region)
plt.title("New Confirmed COVID-19 Cases Over Time by NHS Region", fontsize=16)
plt.xlabel("Date", fontsize=14)
plt.ylabel("New Confirmed Cases", fontsize=14)
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 2. Time Series Graph of Cumulative Confirmed Cases Over Time for Each NHS Region
plt.figure(figsize=(12, 6))
for region in merged_data["region"].unique():
    region_data = merged_data[merged_data["region"] == region]
    plt.plot(region_data["date"], region_data["cumulative_confirmed"], label=region)
plt.title("Cumulative Confirmed COVID-19 Cases Over Time by NHS Region", fontsize=16)
plt.xlabel("Date", fontsize=14)
plt.ylabel("Cumulative Confirmed Cases", fontsize=14)
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 3. Time Series Graph of New Admissions Over Time for Each NHS Region
plt.figure(figsize=(12, 6))
for region in merged_data["region"].unique():
    region_data = merged_data[merged_data["region"] == region]
    plt.plot(region_data["date"], region_data["newAdmissions"], label=region)
plt.title("New Admissions Over Time by NHS Region", fontsize=16)
plt.xlabel("Date", fontsize=14)
plt.ylabel("New Admissions", fontsize=14)
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 4. New Confirmed COVID-19 Cases per 100,000 Population Over Time by NHS Region
population_data = merged_data[["region", "population"]].drop_duplicates()
for region in merged_data["region"].unique():
    region_data = merged_data[merged_data["region"] == region]
    region_population = population_data[population_data["region"] == region]["population"].values[0]
    merged_data.loc[merged_data["region"] == region, "new_confirmed_per_100k"] = (
        region_data["new_confirmed"] / region_population
    ) * 100000
    merged_data.loc[merged_data["region"] == region, "new_deceased_per_100k"] = (
        region_data["new_deceased"] / region_population
    ) * 100000

plt.figure(figsize=(12, 6))
for region in merged_data["region"].unique():
    region_data = merged_data[merged_data["region"] == region]
    plt.plot(region_data["date"], region_data["new_confirmed_per_100k"], label=region)

lockdown_periods = [
    ("2020-03-23", "2020-05-10"),
    ("2020-11-05", "2020-12-02"),
    ("2021-01-06", "2021-03-08"),
]

for start, end in lockdown_periods:
    plt.axvspan(start, end, color="purple", alpha=0.2)

plt.title("New Confirmed COVID-19 Cases per 100,000 Population Over Time by NHS Region", fontsize=16)
plt.xlabel("Date", fontsize=14)
plt.ylabel("New Confirmed Cases per 100,000 Population", fontsize=14)
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 5. Scatter Plot: New Confirmed Cases vs. New Admissions
plt.figure(figsize=(12, 6))
sns.scatterplot(data=merged_data, x="new_confirmed", y="newAdmissions", hue="region", style="region")
plt.title("New Confirmed Cases vs. New Admissions by NHS Region", fontsize=16)
plt.xlabel("New Confirmed Cases", fontsize=14)
plt.ylabel("New Admissions", fontsize=14)
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()

# 6. Heat Map of New Confirmed Cases Over Time by Region
# Create a pivot table
feature = "new_confirmed"
pivot_cases_time_region = merged_data.pivot_table(
    index="date", columns="region", values=feature, aggfunc="sum"
).fillna(0)


# Format the dates to only show 
pivot_cases_time_region.index = pivot_cases_time_region.index.strftime("%Y-%m")


# Create the heatmap
plt.figure(figsize=(10, 8))  # Adjusted for a more compact display
sns.heatmap(pivot_cases_time_region, cmap="viridis")
plt.title(f"Heat Map of {feature} COVID-19 Cases Over Time by NHS Region", fontsize=14, loc="center", pad=20)
plt.xlabel("NHS Region", fontsize=12)
plt.ylabel("Date", fontsize=12)
plt.xticks(rotation=45, fontsize=10)
plt.yticks(fontsize=10)
plt.grid(False)  # Remove gridlines for better visibility
plt.tight_layout()  # Ensures everything fits without overlap
plt.savefig(f'../../reports/images/heatmap_{feature}_cases.pdf')
plt.show()


# 5. Box Plot for Hospital Cases by Region
sns.boxplot(x="region", y="hospitalCases", data=merged_data)
plt.title("Distribution of Hospital Cases by NHS Region")
plt.xlabel("NHS Region")
plt.ylabel("Hospital Cases")
plt.xticks(rotation=45)
plt.show()

# 6. Correlation Heat Map
# Select relevant metrics for correlation
metrics_for_correlation = merged_data[
    [
        "new_confirmed",
        "new_deceased",
        "hospitalCases",
        "newAdmissions",
        "cumulative_confirmed",
        "cumulative_deceased",
    ]
]
correlation_matrix = metrics_for_correlation.corr()

sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heat Map of COVID-19 Metrics")
plt.show()


weekly_data = merged_data.groupby(['region', pd.Grouper(key='date', freq='W')]).agg({
    'covidOccupiedMVBeds': 'mean',  # Average number of ICU beds occupied during the week
    'newAdmissions': 'sum', # Total new admissions during the week
    'new_confirmed': 'sum', # Total new confirmed cases during the week
    'new_deceased': 'sum', # Total new deceased during the week
    'hospitalCases': 'mean', # Average number of hospital cases during the wee
}).reset_index()


region = weekly_data['region'].unique()


for area in region:
    area_data = weekly_data[weekly_data['region'] == area]
    plt.plot(area_data['date'], area_data['covidOccupiedMVBeds'], label=f"{area} - ICU Beds")
    
for start, end in lockdown_periods:
    plt.axvspan(start, end, color="purple", alpha=0.3, label='Lockdown Periods' if start == '2020-03-23' else "")

plt.title('Weekly Trends of ICU Bed Occupancy Across Selected Regions')
plt.xlabel('Date')
plt.ylabel('Average ICU Beds Occupied')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
# plt.savefig('../../images/weekly_icu_beds_occupancy.pdf')
plt.show()


from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

cluster_data = merged_data.groupby(['region', 'date']).agg({
    'new_confirmed': 'mean',
    'new_deceased': 'mean',
    'hospitalCases': 'mean',
    'newAdmissions': 'mean',
    'covidOccupiedMVBeds': 'mean'
}).reset_index()

scaler = StandardScaler()
cluster_features = scaler.fit_transform(cluster_data[['new_confirmed', 'new_deceased', 'hospitalCases', 'newAdmissions', 'covidOccupiedMVBeds']])

silhouette_scores = []
for k in range(2, 21):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(cluster_features)
    score = silhouette_score(cluster_features, kmeans.labels_)
    silhouette_scores.append(score)
    
    
plt.plot(range(2, 21), silhouette_scores, marker='o')
plt.title('Silhouette Score for Different Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.show()

kmeans = KMeans(n_clusters=5, random_state=42)
cluster_data['cluster'] = kmeans.fit_predict(cluster_features)

# plot the clusters on a scatter plot based on all the features
plt.figure(figsize=(12, 8))
sns.scatterplot(data=cluster_data, x='new_confirmed', y='new_deceased', hue='cluster', style='cluster', palette='viridis')
plt.title('Clusters of Regions based on COVID-19 Metrics')
plt.xlabel('New Confirmed Cases')
plt.ylabel('New Deceased Cases')
plt.legend(title='Cluster')
plt.tight_layout()
plt.show()