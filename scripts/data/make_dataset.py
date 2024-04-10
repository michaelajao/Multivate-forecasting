import pandas as pd
from epiweeks import Week
from datetime import datetime
import requests
from pathlib import Path
import os
import logging
import sys

# Function Definitions
def get_cdc_epiweek(date):
    """Convert a date to the corresponding CDC epidemiological week."""
    week = Week.fromdate(date)
    return week.cdcformat()


def save_dataframe(df, path, format='pickle'):
    """Save DataFrame to a specified path in either pickle or CSV format."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if format == 'pickle':
        df.to_pickle(path)
    elif format == 'csv':
        df.to_csv(path)
    else:
        raise ValueError("Unsupported format. Use 'pickle' or 'csv'.")


def filter_data(df, start_date):
    """Filter DataFrame based on a start date."""
    return df[df["date"] >= start_date]


def aggregate_weekly_data(df):
    """Aggregate data on a weekly basis by region and epidemiological week."""
    return (
        df.groupby(["region", "epi_week"])
        .agg(
            {
                "date": "first",
                "new_confirmed": "sum",
                "new_deceased": "sum",
                "cumulative_confirmed": "max",
                "cumulative_deceased": "max",
                "population": "first",
            }
        )
        .reset_index()
    )


def display_data_info(*dfs):
    """Display basic information about DataFrames."""
    for df in dfs:
        df.info()
        print("\n")


def find_date_range(df):
    """Find the start and end date of the DataFrame."""
    start_date = df["date"].min()
    end_date = df["date"].max()
    return start_date, end_date


def clean_and_adjust_df(df):
    """Clean and adjust the DataFrame by dropping nulls and adjusting 'region'."""
    df.dropna(inplace=True)  # Drop rows with any null values
    df["region"] = df["region"].str.replace(", England", "")
    return df


def download_data(urls):
    """Download data from a list of URLs and save as CSV files."""
    for idx, url in enumerate(urls, start=1):
        response = requests.get(url)
        if response.status_code == 200:
            content = response.content
            filename = f"../../data/raw/covid_data_{idx}.csv"
            with open(filename, "wb") as csv_file:
                csv_file.write(content)
            print(f"Data from URL {idx} saved as '{filename}'")
        else:
            print(f"Failed to retrieve data from URL {idx}")


def load_data():
    """Load the data from CSV files into a list of DataFrames."""
    nhs_region_dfs = []
    for idx in range(1, 8):
        filename = f"../../data/raw/covid_data_{idx}.csv"
        nhs_region_df = pd.read_csv(filename)
        nhs_region_dfs.append(nhs_region_df)
    return nhs_region_dfs


def preprocess_data(df, start_date, end_date):
    """Preprocess the data by filtering and filling missing values."""
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
    df = df.fillna(0)
    return df


def main():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # URLs for the CSV files of different regions
    urls = [
        "https://storage.googleapis.com/covid19-open-data/v3/location/GB_UKC.csv",
        "https://storage.googleapis.com/covid19-open-data/v3/location/GB_UKD.csv",
        "https://storage.googleapis.com/covid19-open-data/v3/location/GB_UKE.csv",
        "https://storage.googleapis.com/covid19-open-data/v3/location/GB_UKF.csv",
        "https://storage.googleapis.com/covid19-open-data/v3/location/GB_UKG.csv",
        "https://storage.googleapis.com/covid19-open-data/v3/location/GB_UKH.csv",
        "https://storage.googleapis.com/covid19-open-data/v3/location/GB_UKI.csv",
        "https://storage.googleapis.com/covid19-open-data/v3/location/GB_UKJ.csv",
        "https://storage.googleapis.com/covid19-open-data/v3/location/GB_UKK.csv",
    ]

    try:
        # Efficient concatenation of dataframes from URLs
        combined_df = pd.concat((pd.read_csv(url) for url in urls), ignore_index=True)
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        sys.exit(1)

    # Select relevant columns
    selected_columns = [
        "date",
        "new_confirmed",
        "new_deceased",
        "cumulative_confirmed",
        "cumulative_deceased",
        "population",
        "openstreetmap_id",
        "latitude",
        "longitude",
        "location_key",
        "subregion1_name"
    ]
    final_df = combined_df[selected_columns].copy()

    # Replace 'location_key' with the actual region name and drop unnecessary columns
    location_to_region = final_df.set_index("location_key")["subregion1_name"].to_dict()
    final_df.loc[:, "region"] = final_df["location_key"].map(location_to_region)
    final_df.drop(["location_key", "subregion1_name"], axis=1, inplace=True)

    # Convert 'date' to datetime and add 'epi_week'
    final_df.loc[:, "date"] = pd.to_datetime(final_df["date"])
    final_df.loc[:, "epi_week"] = final_df["date"].apply(get_cdc_epiweek)

    # Data Cleaning and Adjusting
    final_df = clean_and_adjust_df(final_df)

    # Filtering Data
    start_date = datetime(2020, 4, 8)
    filtered_df = filter_data(final_df, start_date)

    # Aggregating Weekly Data
    weekly_aggregated_df = aggregate_weekly_data(filtered_df)

    # Saving Data
    save_dataframe(final_df, "../../data/raw/pickle/covid19_data.pkl")
    save_dataframe(filtered_df, "../../data/raw/pickle/covid19_data_from_april_8.pkl")
    save_dataframe(
        weekly_aggregated_df,
        "../../data/raw/pickle/covid19_weekly_data_by_region_from_april_8.pkl",
    )

    save_dataframe(final_df, "../../data/raw/csv/covid19_data.csv")
    save_dataframe(filtered_df, "../../data/raw/csv/covid19_data_from_april_8.csv")
    save_dataframe(
        weekly_aggregated_df,
        "../../data/raw/csv/covid19_weekly_data_by_region_from_april_8.csv",
    )

    # Displaying Date Range and DataFrame Information
    for df in [final_df, filtered_df, weekly_aggregated_df]:
        start_date, end_date = find_date_range(df)
        print(f"Date Range: {start_date} to {end_date}")
        display_data_info(df)

    # List of URLs
    urls = [
        "https://api.coronavirus.data.gov.uk/v2/data?areaType=nhsRegion&areaCode=E40000007&metric=covidOccupiedMVBeds&metric=hospitalCases&metric=newAdmissions&metric=cumAdmissions&format=csv",
        "https://api.coronavirus.data.gov.uk/v2/data?areaType=nhsRegion&areaCode=E40000003&metric=covidOccupiedMVBeds&metric=hospitalCases&metric=newAdmissions&metric=cumAdmissions&format=csv",
        "https://api.coronavirus.data.gov.uk/v2/data?areaType=nhsRegion&areaCode=E40000008&metric=covidOccupiedMVBeds&metric=hospitalCases&metric=newAdmissions&metric=cumAdmissions&format=csv",
        "https://api.coronavirus.data.gov.uk/v2/data?areaType=nhsRegion&areaCode=E40000009&metric=covidOccupiedMVBeds&metric=hospitalCases&metric=newAdmissions&metric=cumAdmissions&format=csv",
        "https://api.coronavirus.data.gov.uk/v2/data?areaType=nhsRegion&areaCode=E40000010&metric=covidOccupiedMVBeds&metric=hospitalCases&metric=newAdmissions&metric=cumAdmissions&format=csv",
        "https://api.coronavirus.data.gov.uk/v2/data?areaType=nhsRegion&areaCode=E40000005&metric=covidOccupiedMVBeds&metric=hospitalCases&metric=newAdmissions&metric=cumAdmissions&format=csv",
        "https://api.coronavirus.data.gov.uk/v2/data?areaType=nhsRegion&areaCode=E40000006&metric=covidOccupiedMVBeds&metric=hospitalCases&metric=newAdmissions&metric=cumAdmissions&format=csv"
    ]

    download_data(urls)

    # Load the data into a DataFrame
    nhs_region_dfs = load_data()

    # Concatenate the DataFrames
    nhs_region_combined_df = pd.concat(nhs_region_dfs, ignore_index=True)

    start_date = pd.to_datetime('2020-04-01')
    end_date = pd.to_datetime('2023-05-31')

    nhs_region_combined_df = preprocess_data(nhs_region_combined_df, start_date, end_date)

    # Save the combined DataFrame
    save_dataframe(nhs_region_combined_df, "../../data/interim/pickle/nhs_region_data.pkl")
    save_dataframe(nhs_region_combined_df, "../../data/interim/csv/nhs_region_data.csv")

if __name__ == "__main__":
    main()
