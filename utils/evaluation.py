import scipy

import pandas as pd
import numpy as np


from sklearn.metrics import precision_score, recall_score
from collections import OrderedDict
from scipy import stats

from utils.data_processing import get_average_detection_delay


def classification_report(y_true_l, **kwargs):
    """
    classification report with benchmark results (input classifier names and predictions)
    :param y_true: labels list (should be same lenght as detectors in kwargs)
    :return: filtered data frame
    """
    detector_dict = OrderedDict()
    detector_dict["Perfect Detector"] = y_true_l[0], y_true_l[0]

    for i, (key, value) in enumerate(kwargs.items()):
        assert len(y_true_l[i]) == len(value)
        detector_dict[key] = value, y_true_l[i]

    detector_dict["Null Detector 1"] = [0] * len(y_true_l[0]), y_true_l[0]
    detector_dict["Null Detector 2"] = [1] * len(y_true_l[0]), y_true_l[0]

    np.random.seed(0)
    detector_dict["Random Detector"] = np.where(np.random.rand(len(y_true_l[0])) >= 0.5, 1, 0), y_true_l[0]

    data = []

    for key, value in detector_dict.items():
        data.append(
            [
                key,
                round(precision_score(value[1], value[0]), 3),
                round(recall_score(value[1], value[0]), 3),
                round(get_average_detection_delay(value[1], value[0]), 3)
            ]
        )

    return pd.DataFrame(columns=["Detector", "Precision", "TPR", "Average Detection Delay"], data=data)


def classification_report(y_true_l, **kwargs):
    """
    classification report with benchmark results (input classifier names and predictions)
    :param y_true: labels list (should be same lenght as detectors in kwargs)
    :return: filtered data frame
    """
    detector_dict = OrderedDict()
    detector_dict["Perfect Detector"] = y_true_l[0], y_true_l[0]

    for i, (key, value) in enumerate(kwargs.items()):
        assert len(y_true_l[i]) == len(value)
        detector_dict[key] = value, y_true_l[i]

    detector_dict["Null Detector 1"] = [0] * len(y_true_l[0]), y_true_l[0]
    detector_dict["Null Detector 2"] = [1] * len(y_true_l[0]), y_true_l[0]

    np.random.seed(0)
    detector_dict["Random Detector"] = np.where(np.random.rand(len(y_true_l[0])) >= 0.5, 1, 0), y_true_l[0]

    data = []

    for key, value in detector_dict.items():
        data.append(
            [
                key,
                round(precision_score(value[1], value[0]), 3),
                round(recall_score(value[1], value[0]), 3),
                round(get_average_detection_delay(value[1], value[0]), 3)
            ]
        )

    return pd.DataFrame(columns=["Detector", "Precision", "TPR", "Average Detection Delay"], data=data)


def classification_report(y_true_l, **kwargs):
    """
    classification report with benchmark results (input classifier names and predictions)
    :param y_true: labels list (should be same lenght as detectors in kwargs)
    :return: filtered data frame
    """
    detector_dict = OrderedDict()
    detector_dict["Perfect Detector"] = y_true_l[0], y_true_l[0]

    for i, (key, value) in enumerate(kwargs.items()):
        assert len(y_true_l[i]) == len(value)
        detector_dict[key] = value, y_true_l[i]

    detector_dict["Null Detector 1"] = [0] * len(y_true_l[0]), y_true_l[0]
    detector_dict["Null Detector 2"] = [1] * len(y_true_l[0]), y_true_l[0]

    np.random.seed(0)
    detector_dict["Random Detector"] = np.where(np.random.rand(len(y_true_l[0])) >= 0.5, 1, 0), y_true_l[0]

    data = []

    for key, value in detector_dict.items():
        data.append(
            [
                key,
                round(precision_score(value[1], value[0]), 3),
                round(recall_score(value[1], value[0]), 3),
                round(get_average_detection_delay(value[1], value[0]), 3)
            ]
        )

    return pd.DataFrame(columns=["Detector", "Precision", "TPR", "Average Detection Delay"], data=data)


def classification_report(y_true_l, **kwargs):
    """
    classification report with benchmark results (input classifier names and predictions)
    :param y_true: labels list (should be same lenght as detectors in kwargs)
    :return: filtered data frame
    """
    detector_dict = OrderedDict()
    detector_dict["Perfect Detector"] = y_true_l[0], y_true_l[0]

    for i, (key, value) in enumerate(kwargs.items()):
        assert len(y_true_l[i]) == len(value)
        detector_dict[key] = value, y_true_l[i]

    detector_dict["Null Detector 1"] = [0] * len(y_true_l[0]), y_true_l[0]
    detector_dict["Null Detector 2"] = [1] * len(y_true_l[0]), y_true_l[0]

    np.random.seed(0)
    detector_dict["Random Detector"] = np.where(np.random.rand(len(y_true_l[0])) >= 0.5, 1, 0), y_true_l[0]

    data = []

    for key, value in detector_dict.items():
        data.append(
            [
                key,
                round(precision_score(value[1], value[0]), 3),
                round(recall_score(value[1], value[0]), 3),
                round(get_average_detection_delay(value[1], value[0]), 3)
            ]
        )

    return pd.DataFrame(columns=["Detector", "Precision", "TPR", "Average Detection Delay"], data=data)


def classification_report(y_true_l, **kwargs):
    """
    classification report with benchmark results (input classifier names and predictions)
    :param y_true: labels list (should be same lenght as detectors in kwargs)
    :return: filtered data frame
    """
    detector_dict = OrderedDict()
    detector_dict["Perfect Detector"] = y_true_l[0], y_true_l[0]

    for i, (key, value) in enumerate(kwargs.items()):
        assert len(y_true_l[i]) == len(value)
        detector_dict[key] = value, y_true_l[i]

    detector_dict["Null Detector 1"] = [0] * len(y_true_l[0]), y_true_l[0]
    detector_dict["Null Detector 2"] = [1] * len(y_true_l[0]), y_true_l[0]

    np.random.seed(0)
    detector_dict["Random Detector"] = np.where(np.random.rand(len(y_true_l[0])) >= 0.5, 1, 0), y_true_l[0]

    data = []

    for key, value in detector_dict.items():
        data.append(
            [
                key,
                round(precision_score(value[1], value[0]), 3),
                round(recall_score(value[1], value[0]), 3),
                round(get_average_detection_delay(value[1], value[0]), 3)
            ]
        )

    return pd.DataFrame(columns=["Detector", "Precision", "TPR", "Average Detection Delay"], data=data)


def test_noraml_dist(x, alpha=0.05):
  """
  Perform the Shapiro-Wilk test for normality

  :param x: The array containing the sample to be tested
  :param alpha: threshold for rejection of null hypothesis
  """
  # For N > 5000 the W test statistic is accurate but the p-value may not be.
  # The chance of rejecting the null hypothesis when it is true is close to 5% regardless of sample size.
  length = min(len(x), 2500)
  stats, p = scipy.stats.shapiro(x[:length])
  print(f"p-value: {p}")
  if p < alpha:  # null hypothesis: the data was drawn from a normal distribution
      print("The null hypothesis can be rejected")
  else:
      print("The null hypothesis cannot be rejected")