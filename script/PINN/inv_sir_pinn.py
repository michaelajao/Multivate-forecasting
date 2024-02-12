import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_and_preprocess_data(filepath):
    try:
        # Load data from a CSV file
        df = pd.read_csv(filepath)
        
        # Ensure the 'date', 'cumulative_confirmed', and 'cumulative_deceased' columns exist
        required_columns = ['date', 'cumulative_confirmed', 'cumulative_deceased', 'population']
        if not all(column in df.columns for column in required_columns):
            raise ValueError("Missing required columns in the dataset")
        
        # Convert 'date' column to datetime format
        df['date'] = pd.to_datetime(df['date'])
        
        # Calculate the number of days since the start of the dataset
        df['days_since_start'] = (df['date'] - df['date'].min()).dt.days
        
        # Smooth the 'cumulative_confirmed' and 'cumulative_deceased' with a 7-day rolling average
        for col in ['cumulative_confirmed', 'cumulative_deceased']:
            df[col] = df[col].rolling(window=7, min_periods=1).mean().fillna(0).astype(int)
        
        # Calculate recovered cases assuming a fixed recovery period
        recovery_period = 21
        df['recovered'] = df['cumulative_confirmed'].shift(recovery_period) - df['cumulative_deceased'].shift(recovery_period)
        
        # Calculate the number of active cases
        df['active_cases'] = df['cumulative_confirmed'] - df['recovered'] - df['cumulative_deceased']
        
        # Calculate the susceptible population (S(t))
        df['S(t)'] = df['population'] - df['active_cases'] - df['recovered'] - df['cumulative_deceased']
        
        # Fill any remaining missing values with 0
        df.fillna(0, inplace=True)
        
        df = df[df['date'] >= '2020-04-01']
        
        return df
    except FileNotFoundError:
        print("File not found. Please check the filepath and try again.")
    except pd.errors.EmptyDataError:
        print("No data found. Please check the file content.")
    except ValueError as e:
        print(e)


def split_time_series_data(df, train_size=0.7, val_size=0.15, test_size=0.15):
    """
    Splits the DataFrame into training, validation, and test sets while maintaining the time series order.

    Args:
        df (pd.DataFrame): The input DataFrame with time series data.
        train_size (float): Proportion of the dataset to allocate to training.
        val_size (float): Proportion of the dataset to allocate to validation.
        test_size (float): Proportion of the dataset to allocate to testing.
    
    Returns:
        tuple: Three DataFrames corresponding to the training, validation, and test sets.
    """
    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError("train_size, val_size, and test_size should sum to 1.")

    n = len(df)
    train_end = int(n * train_size)
    val_end = train_end + int(n * val_size)

    train_data = df.iloc[:train_end]
    val_data = df.iloc[train_end:val_end]
    test_data = df.iloc[val_end:]

    return train_data, val_data, test_data


df = load_and_preprocess_data("../../data/region_daily_data/East Midlands.csv") 


# Use the function to split the data
train_df, val_df, test_df = split_time_series_data(df, train_size=0.7, val_size=0.15, test_size=0.15)

# Optionally, select a specific size for training (e.g., first 30 data points from the training set)
train_df_selected = train_df.head(30)

def df_to_tensors(df):
    """
    Converts specified columns of a DataFrame into PyTorch tensors.

    Args:
        df (pd.DataFrame): Input DataFrame with time series data.

    Returns:
        tuple: Tensors for days_since_start, cumulative_confirmed, and cumulative_deceased.
    """

    t_data_tensor = torch.tensor(range(len(df)), dtype=torch.float32).view(-1, 1). requires_grad_(True).to(device)
    I_data_tensor = torch.tensor(df['cumulative_confirmed'].values, dtype=torch.float32).view(-1, 1).to(device)
    R_data_tensor = torch.tensor(df['cumulative_deceased'].values, dtype=torch.float32).view(-1, 1).to(device)

    return t_data_tensor, I_data_tensor, R_data_tensor

# Convert the selected training DataFrame to tensors
train_tensors[2] = df_to_tensors(train_df_selected)
