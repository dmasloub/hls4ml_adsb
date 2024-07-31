import numpy as np
import pandas as pd
from scipy import stats

def rolled(data, window_size):
    """
    Generator to yield batches of rows from a data frame of specified window size.
    
    Parameters:
    data (array-like): Input data from which windows are generated.
    window_size (int): The size of each window.

    Yields:
    array-like: Subsequent windows of the input data.
    """
    count = 0
    while count <= len(data) - window_size:
        yield data[count: count + window_size]
        count += 1

def max_rolled(data, window_size):
    """
    Returns the maximum value for each rolling sliding window.
    
    Parameters:
    data (array-like): List of values from which rolling windows are generated.
    window_size (int): The size of each window.

    Returns:
    np.array: Array of maximum values for each rolling window.
    """
    max_values = []
    for window in rolled(data, window_size):
        max_values.append(max(window))
    
    return np.array(max_values)

def get_windows_data(data_frame, labels, window_size, tsfresh=True):
    """
    Prepare data for autoencoder and tsfresh processing.
    
    Parameters:
    data_frame (pd.DataFrame): Input data frame containing features.
    labels (array-like): Corresponding labels for the data.
    window_size (int): The size of each window.
    tsfresh (bool): Indicator whether to prepare dataframe for tsfresh (add 'id' and 'time' columns).

    Returns:
    tuple: A tuple (X, y) where X is the processed data and y are the corresponding labels.
    """
    all_windows = []

    # Iterate over windows generated from the input data frame
    for index, window in enumerate(rolled(data_frame, window_size)):
        window = window.copy()
        if tsfresh:
            window['id'] = [index] * window.shape[0]
            window['time'] = list(range(window.shape[0]))
        all_windows.append(window)

    # Combine all windows into a single data frame or array
    if all_windows:
        X = pd.concat(all_windows, ignore_index=True) if tsfresh else np.array([w.values for w in all_windows])
    else:
        X = pd.DataFrame() if tsfresh else np.array([])

    # Compute the max rolled values for the labels
    y = max_rolled(labels, window_size)

    return X, y

def get_average_detection_delay(y_true, y_pred):
    """
    Calculate the average detection delay for true labels and predictions.
    
    Parameters:
    y_true (array-like): True labels.
    y_pred (array-like): Predicted labels.
    
    Returns:
    float: Average detection delay.
    """
    assert len(y_true) == len(y_pred), "The lengths of true and predicted labels must be equal."

    in_window = False
    detected_in_window = False
    detection_delay_sum = 0
    windows_count = 0

    for i in range(len(y_true) - 1):
        curr_true = y_true[i]
        next_true = y_true[i + 1]
        curr_pred = y_pred[i]

        # If in a detection window and event not detected, increment delay
        if in_window and not detected_in_window:
            if curr_pred == 1:
                detected_in_window = True
            else:
                detection_delay_sum += 1

        # Check for start of a new event window
        if (curr_true == 0 and next_true == 1) or (curr_true == 1 and i == 0):
            in_window = True
            windows_count += 1

        # Check for end of an event window
        if curr_true == 1 and next_true == 0:
            in_window = False
            detected_in_window = False

    # Adjust for windows not padded at the end or beginning
    if y_true[-1] == 1:
        detection_delay_sum += 1
    if y_true[0] == 1:
        detection_delay_sum += 1

    return detection_delay_sum / windows_count if windows_count > 0 else 0

def anomaly_score(score, mu, sig):
    """
    Calculate anomaly scores based on the CDF of a normal distribution.
    
    Parameters:
    score (array-like): List of scores given by the model.
    mu (float): Mean of the normal distribution.
    sig (float): Standard deviation of the normal distribution.
    
    Returns:
    array-like: Anomaly scores.
    """
    # Input validation
    if not isinstance(score, (list, np.ndarray)):
        raise ValueError("score must be a list or array-like")
    if not isinstance(mu, (int, float)):
        raise ValueError("mu must be a numeric value")
    if not isinstance(sig, (int, float)):
        raise ValueError("sig must be a numeric value")

    return 1 - stats.norm.sf(score, mu, sig)

def q_verdict(x, mu, sig, n=0.1):
    """
    Provide a verdict on anomaly based on the CDF of a normal distribution.
    
    Parameters:
    x (array-like): List of scores given by the model.
    mu (float): Mean of the normal distribution.
    sig (float): Standard deviation of the normal distribution.
    n (float): Threshold for anomaly detection (default is 0.1).
    
    Returns:
    np.ndarray: Array of verdicts (1 for anomaly, 0 for normal).
    """
    # Input validation
    if not isinstance(x, (list, np.ndarray, pd.Series)):
        raise ValueError("x must be a list or array-like")
    if not isinstance(mu, (int, float)):
        raise ValueError("mu must be a numeric value")
    if not isinstance(sig, (int, float)):
        raise ValueError("sig must be a numeric value")
    if not isinstance(n, (int, float)):
        raise ValueError("n must be a numeric value")

    # Ensure x is a NumPy array
    x = np.asarray(x)

    anomaly_scores = anomaly_score(x, mu, sig)
    return np.where(anomaly_scores >= 1 - n, 1, 0)