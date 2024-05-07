import os
    """
    The code provided loads, merges, preprocesses, and visualizes COVID-19 data for NHS regions,
    including time series graphs, scatter plots, heat maps, box plots, correlation analysis, weekly
    trends, and clustering analysis.
    
    :param file_path: The `file_path` parameter in the code you provided is a string variable that
    stores the path to the directory where the data files are located. It is used in functions like
    `load_and_prepare_data` to construct the full path to the data files by joining it with the
    `filename`
    :param filename: The `filename` parameter in the code you provided is a variable used to store the
    name of the file being loaded and prepared as data. It is used as an input to the
    `load_and_prepare_data` function to specify the name of the file to be read and processed
    :param date_col: The `date_col` parameter is used to specify the column in the dataset that contains
    date information. In the provided code, the `date_col` parameter is used in functions like
    `load_and_prepare_data` and `map_regions_and_merge` to properly format and manipulate date-related
    data. It helps
    :return: The code provided returns visualizations and analysis based on the COVID-19 data for NHS
    regions. Here is a summary of what is being returned:
    """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Set the default style for plots
plt.style.use("fivethirtyeight")
plt.rcParams.update(
    {
        "lines.linewidth": 2,
        "font.family": "serif",
        "axes.titlesize": 20,
        "axes.labelsize": 14,
        "figure.figsize": [15, 8],
        "figure.autolayout": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "grid.color": "0.75",
        "legend.fontsize": "medium",
    }
)

# Constants
data_path = "../../data/hos_data/"
filtered_data_file = "filtered_data.csv"
covid19_data_file = "covid19_data_from_april_8.csv"
date_column = "date"
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


def load_and_prepare_data(file_path, filename, date_col):
    """Load data from CSV file and prepare it with proper date formatting."""
    full_path = os.path.join(file_path, filename)
    try:
        df = pd.read_csv(full_path)
        df[date_col] = pd.to_datetime(df[date_col])
        return df
    except Exception as e:
        print(f"Error loading or preparing data: {e}")
        return pd.DataFrame()  # Return empty DataFrame on error


def map_regions_and_merge(df1, df2, mapping, merge_columns, drop_columns):
    """Map regions in one dataframe to another and merge them."""
    df2["mapped_region"] = df2["region"].map(mapping)
    merged_df = pd.merge(
        df1, df2, how="inner", left_on=merge_columns, right_on=["date", "mapped_region"]
    )
    merged_df.drop_duplicates(inplace=True)
    merged_df.drop(columns=drop_columns, inplace=True, errors="ignore")
    return merged_df


# Load and prepare data
filtered_data = load_and_prepare_data(data_path, filtered_data_file, date_column)
covid19_data = load_and_prepare_data(data_path, covid19_data_file, date_column)

# Merge dataframes
merge_columns = ["date", "areaName"]
drop_columns = ["Unnamed: 0", "mapped_region", "areaType", "areaCode"]
merged_data = map_regions_and_merge(
    filtered_data, covid19_data, region_mapping, merge_columns, drop_columns
)

# Saving merged data
merged_data.to_csv(os.path.join(data_path, "merged_data.csv"))
merged_data.to_pickle(os.path.join(data_path, "merged_data.pkl"))



merged_data["date"] = pd.to_datetime(merged_data["date"])
# 1. Time Series Graph of New Confirmed Cases Over Time for Each NHS Region
for region in merged_data["region"].unique():
    region_data = merged_data[merged_data["region"] == region]
    plt.plot(region_data["date"], region_data["new_confirmed"], label=region)

plt.title("New Confirmed COVID-19 Cases Over Time by NHS Region")
plt.xlabel("Date")
plt.ylabel("New Confirmed Cases")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# plt.figure(figsize=(15, 10))
for region in merged_data["region"].unique():
    region_data = merged_data[merged_data["region"] == region]
    plt.plot(region_data["date"], region_data["cumulative_confirmed"], label=region)

plt.title("Cumulative Confirmed COVID-19 Cases Over Time by NHS Region")
plt.xlabel("Date")
plt.ylabel("Cumulative Confirmed Cases")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 2. Time Series Graph of New Admissions Over Time for Each NHS Region
for region in merged_data["region"].unique():
    region_data = merged_data[merged_data["region"] == region]
    plt.plot(region_data["date"], region_data["newAdmissions"], label=region)

plt.title("New Admissions Over Time by NHS Region")
plt.xlabel("Date")
plt.ylabel("New Admissions")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

population_data = merged_data[["region", "population"]].drop_duplicates()
# preprocess the data for the new cases/deaths per 100,000 population for each region
for region in merged_data["region"].unique():
    region_data = merged_data[merged_data["region"] == region]
    region_population = population_data[population_data["region"] == region][
        "population"
    ].values[0]
    merged_data.loc[merged_data["region"] == region, "new_confirmed_per_100k"] = (
        region_data["new_confirmed"] / region_population
    ) * 100000
    merged_data.loc[merged_data["region"] == region, "new_deceased_per_100k"] = (
        region_data["new_deceased"] / region_population
    ) * 100000


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


plt.title("New Confirmed COVID-19 Cases per 100,000 Population Over Time by NHS Region")
plt.xlabel("Date")
plt.ylabel("New Confirmed Cases per 100,000 Population")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
# plt.savefig("../../images/new_confirmed_per_100k.pdf")
plt.show()


# 3. Scatter Plot: New Confirmed Cases vs. New Admissions
sns.scatterplot(
    data=merged_data, x="new_confirmed", y="newAdmissions", hue="region", style="region"
)
plt.title("New Confirmed Cases vs. New Admissions by NHS Region")
plt.xlabel("New Confirmed Cases")
plt.ylabel("New Admissions")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()

# 4. Heat Map of New Confirmed Cases Over Time by Region
# Pivot data for heat map
pivot_cases_time_region = merged_data.pivot_table(
    index="date", columns="region", values="new_confirmed", aggfunc="sum"
).fillna(0)

sns.heatmap(pivot_cases_time_region, cmap="viridis")
plt.title("Heat Map of New Confirmed COVID-19 Cases Over Time by NHS Region")
plt.xlabel("NHS Region")
plt.ylabel("Date")
plt.xticks(rotation=45)
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