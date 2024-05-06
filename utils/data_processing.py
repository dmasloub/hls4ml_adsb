import numpy as np
import pandas as pd
from scipy import stats


def rolled(list_, window_size):
    """
    generator to yield batches of rows from a data frame of <window_size>
    :param list_: array like object
    :param window_size: window size
    :return: batch of rows
    """
    count = 0
    while count <= len(list_) - window_size:
        yield list_[count: count + window_size]
        count += 1


def max_rolled(list_, window_size):
    """
    returns max value for each rolling sliding window
    :param list_: list of values
    :param window_size: window size for each instance
    :return:  max value for each rolling sliding window
    """
    y = []
    for val in rolled(list_, window_size):
        y.append(max(val))

    return np.array(y)


def get_windows_data(df, labels, window_size, tsfresh=True):
    """
    get data for autoencoder (batches of window sized data) and tsfresh phase
    :param df: data frame
    :param labels: labels
    :param window_size: window size for each instance
    :param tsfresh: indicator whether to prepare dataframe for tsfresh (add 'id' and 'time' column)
    :return: (X, y)
    """
    all_windows = []

    for i, window in enumerate(rolled(df, window_size)):
        window = window.copy()
        if tsfresh:
            window['id'] = [i] * window.shape[0]
            window['time'] = list(range(window.shape[0]))
        all_windows.append(window)

    if all_windows:
        X = pd.concat(all_windows, ignore_index=True) if tsfresh else np.array([w.values for w in all_windows])
    else:
        X = pd.DataFrame() if tsfresh else np.array([])

    y = max_rolled(labels, window_size)

    return X, y


def diff_data(df, cols, lag, order):
  """
  apply time series differrencing to the data
  :param df: dataframe
  :param cols: columns to apply differencing
  :param lag: k-lag
  :param order: k-order
  :return: df_diff - dataframe with differencing
  """
  assert lag > 0
  assert order > 0

  df_diff = df[cols]

  for i in range(order):
      df_diff = df_diff.diff(periods=lag)

      # remove NAN value rows from df_diff
      df_diff = df_diff[lag:]

  # return excluded columns
  excluded_cols = [x for x in df.columns if x not in cols]

  for col in excluded_cols:
    df_diff[col] = df[col][lag * order:]

  return df_diff


def filter_outliers(df, std=5, cols=None):
  """
  remove extreme outliers in data
  :param df: data (data frame)
  :param std: amount of standard deviation (remove values withe larger values)
  :param df: columns to apply
  :return: filtered data frame
  """
  selected_cols = df.columns if cols is None else cols

  return df[(np.abs(stats.zscore(df[selected_cols])) < std).all(axis=1)]


def get_average_detection_delay(y_true, y_pred):
  """
  return average detection delay for labels and predections
  :param y_true: labels
  :param y_pred: predections
  :return: average detection delay
  """
  assert len(y_true) == len(y_pred)

  in_window = False
  detected_in_window = False
  detection_delay_sum = 0
  windows_count = 0

  for i in range(len(y_true) - 1):
    curr_true = y_true[i]
    next_true = y_true[i + 1]
    curr_pred = y_pred[i]

    if in_window and not detected_in_window:
      if curr_pred == 1:
        detected_in_window = True
      else:
        detection_delay_sum += 1

    if (curr_true == 0 and next_true == 1) or (curr_true == 1 and i == 0):
      in_window = True
      windows_count += 1

    if curr_true == 1 and next_true == 0:
      in_window = False
      detected_in_window = False

  # window is not padded
  if next_true == 1:
    detection_delay_sum += 1

  # window is not padded
  if y_true[0] == 1:
    detection_delay_sum += 1

  return detection_delay_sum / windows_count if windows_count > 0 else 0