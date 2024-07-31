import numpy as np
import pandas as pd
from collections import OrderedDict
from sklearn.metrics import precision_score, recall_score, accuracy_score
from src.utils.utils import get_average_detection_delay

def classification_report(y_true_l, **kwargs):
    """
    Generate a classification report with benchmark results.
    
    Parameters:
    y_true_l (list): List of true labels.
    kwargs (dict): Dictionary of classifier names and their predictions.
    
    Returns:
    pd.DataFrame: DataFrame containing the classification report with metrics for each detector.
    """
    detector_dict = OrderedDict()
    
    # Adding Perfect Detector for benchmark comparison
    detector_dict["Perfect Detector"] = y_true_l[0], y_true_l[0]

    # Adding each classifier's predictions and true labels to the dictionary
    for i, (key, value) in enumerate(kwargs.items()):
        assert len(y_true_l[i]) == len(value), "Length of true labels and predictions must be equal"
        detector_dict[key] = value, y_true_l[i]

    # Adding Null Detectors (always predicting 0 or 1)
    detector_dict["Null Detector 1"] = [0] * len(y_true_l[0]), y_true_l[0]
    detector_dict["Null Detector 2"] = [1] * len(y_true_l[0]), y_true_l[0]

    # Adding Random Detector
    np.random.seed(0)
    detector_dict["Random Detector"] = np.where(np.random.rand(len(y_true_l[0])) >= 0.5, 1, 0), y_true_l[0]

    data = []

    # Calculating precision, recall, and average detection delay for each detector
    for key, (pred, true) in detector_dict.items():
        precision = round(precision_score(true, pred), 3)
        recall = round(recall_score(true, pred), 3)
        avg_delay = round(get_average_detection_delay(true, pred), 3)
        data.append([key, precision, recall, avg_delay])
  
    return pd.DataFrame(columns=["Detector", "Precision", "Recall", "Average Detection Delay"], data=data)