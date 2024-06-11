import pandas as pd
import numpy as np
import os

# File paths
path = "../../data/interim/pickle/nhs_region_data.pkl"
path2 = "../../data/raw/pickle/covid19_data.pkl"

# Load data
data1 = pd.read_pickle(path)
data2 = pd.read_pickle(path2)

# Convert date columns to datetime
data1["date"] = pd.to_datetime(data1["date"])
data2["date"] = pd.to_datetime(data2["date"])

# Region mapping
region_mapping = {
    "East of England": "East of England",
    "London Region": "London",
    "West Midlands": "Midlands",
    "East Midlands": "Midlands",
    "North East England": "North East and Yorkshire",
    "Yorkshire and the Humber": "North East and Yorkshire",
    "North West England": "North West",
    "South East England": "South East",
    "South West England": "South West",
}

# Map regions in data2
data2["region"] = data2["region"].map(region_mapping)

# Check for NaN values after mapping
if data2["region"].isna().any():
    print("There are unmapped regions in data2 after applying region_mapping.")
    print(data2[data2["region"].isna()]["region"].unique())

# Handle duplicates in data2 by aggregating numerical columns
numeric_columns_data2 = data2.select_dtypes(include=[np.number]).columns.tolist()
data2_aggregated = data2.groupby(['region', 'date'])[numeric_columns_data2].sum().reset_index()

# Merge the data based on region and date
merged_data = data1.merge(data2_aggregated, left_on=["areaName", "date"], right_on=["region", "date"], how="inner")

# Verify the uniqueness of the merged data
duplicates_exist = merged_data.duplicated(subset=["areaName", "date"]).any()
duplicates = merged_data[merged_data.duplicated(subset=["areaName", "date"], keep=False)]

if duplicates_exist:
    print("There are duplicates in the merged data based on 'areaName' and 'date'.")
    print(duplicates.head(10))  # Display a sample of the duplicates

# Aggregate duplicates by summing the numerical columns in merged data
numeric_columns = merged_data.select_dtypes(include=[np.number]).columns.tolist()
aggregated_data = merged_data.groupby(['areaName', 'date'])[numeric_columns].sum().reset_index()

# Ensure target directory exists
os.makedirs("../../data/processed", exist_ok=True)

# Save the cleaned and aggregated data
aggregated_data.to_pickle("../../data/processed/merged_nhs_covid_data.pkl")
aggregated_data.to_csv("../../data/processed/merged_nhs_covid_data.csv", index=False)

# Aggregate data to create England-wide data
england_data = aggregated_data.groupby("date")[numeric_columns].sum().reset_index()

# Add areaName column for England
england_data["areaName"] = "England"

# Save aggregated England data
england_data.to_pickle("../../data/processed/england_data.pkl")
england_data.to_csv("../../data/processed/england_data.csv", index=False)