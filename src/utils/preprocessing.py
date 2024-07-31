import pandas as pd
import numpy as np
from scipy import stats

def filter_outliers(df, std=5, cols=None):
    """
    Remove extreme outliers in the data based on z-score.
    
    Parameters:
    df (pd.DataFrame): Input data frame.
    std (int): Number of standard deviations to use as the threshold. Values beyond this threshold are considered outliers.
    cols (list): List of columns to apply outlier filtering. If None, all columns are used.
    
    Returns:
    pd.DataFrame: Filtered data frame with outliers removed.
    """
    # Select columns to apply outlier filtering
    selected_cols = df.columns if cols is None else cols

    # Calculate the z-score for the selected columns
    z_scores = np.abs(stats.zscore(df[selected_cols]))

    # Filter out rows with z-scores beyond the specified number of standard deviations
    filtered_df = df[(z_scores < std).all(axis=1)]

    return filtered_df

def diff_data(df, cols, lag, order):
    """
    Apply time series differencing to the data.
    
    Parameters:
    df (pd.DataFrame): Input data frame.
    cols (list): Columns to apply differencing.
    lag (int): Number of periods to lag for differencing.
    order (int): Number of differencing steps to apply.
    
    Returns:
    pd.DataFrame: Data frame with differenced data.
    """
    # Ensure lag and order are positive integers
    assert lag > 0, "Lag must be greater than 0"
    assert order > 0, "Order must be greater than 0"

    # Apply differencing to specified columns
    df_diff = df[cols].copy()
    
    for _ in range(order):
        df_diff = df_diff.diff(periods=lag)
        # Remove NaN value rows resulting from differencing
        df_diff = df_diff[lag:]

    # Include columns that were not differenced
    excluded_cols = [col for col in df.columns if col not in cols]
    
    for col in excluded_cols:
        df_diff[col] = df[col][lag * order:].values

    return df_diff