# src/utils/evaluation.py

import numpy as np
import pandas as pd
from collections import OrderedDict
from sklearn.metrics import precision_score, recall_score
from scipy import stats
from typing import List, Dict, Any

from src.utils.common_utils import CommonUtils
from src.utils.logger import Logger


class EvaluationUtils:
    @staticmethod
    def get_average_detection_delay(y_true: List[int], y_pred: List[int]) -> float:
        """
        Calculate the average detection delay for true labels and predictions.

        Parameters:
            y_true (List[int]): True binary labels.
            y_pred (List[int]): Predicted binary labels.

        Returns:
            float: Average detection delay.
        """
        logger = Logger.get_logger(__name__)
        try:
            assert len(y_true) == len(y_pred), "The lengths of true and predicted labels must be equal."

            in_window = False
            detected_in_window = False
            detection_delay_sum = 0
            windows_count = 0

            for i in range(len(y_true) - 1):
                curr_true = y_true[i]
                next_true = y_true[i + 1]
                curr_pred = y_pred[i]

                # If currently in a detection window and event not yet detected
                if in_window and not detected_in_window:
                    if curr_pred == 1:
                        detected_in_window = True
                    else:
                        detection_delay_sum += 1

                # Check for start of a new event window
                if (curr_true == 0 and next_true == 1) or (curr_true == 1 and i == 0):
                    in_window = True
                    detected_in_window = False
                    windows_count += 1

                # Check for end of an event window
                if curr_true == 1 and next_true == 0:
                    in_window = False
                    detected_in_window = False

            # Handle cases where the event window extends to the end or starts at the beginning
            if y_true[-1] == 1:
                detection_delay_sum += 1
            if y_true[0] == 1 and len(y_true) > 1:
                detection_delay_sum += 1

            average_delay = detection_delay_sum / windows_count if windows_count > 0 else 0
            logger.debug(f"Average Detection Delay calculated: {average_delay}")

            return average_delay

        except Exception as e:
            logger.error(f"Error calculating average detection delay: {e}", exc_info=True)
            raise

    @staticmethod
    def classification_report(y_true_l: List[List[int]], **kwargs: List[int]) -> pd.DataFrame:
        """
        Generate a classification report with benchmark results.

        Parameters:
            y_true_l (List[List[int]]): List containing lists of true labels for each classifier.
            **kwargs (Dict[str, List[int]]): Dictionary of classifier names and their predicted labels.

        Returns:
            pd.DataFrame: DataFrame containing the classification report with metrics for each detector.
        """
        logger = Logger.get_logger(__name__)
        try:
            detector_dict = OrderedDict()

            if not isinstance(y_true_l, list):
                raise ValueError("y_true_l must be a list of true label lists.")

            if len(y_true_l) == 0:
                raise ValueError("y_true_l must contain at least one set of true labels.")

            # Adding Perfect Detector for benchmark comparison
            detector_dict["Perfect Detector"] = (y_true_l[0], y_true_l[0])

            # Adding each classifier's predictions and true labels to the dictionary
            for key, y_pred in kwargs.items():
                if not isinstance(y_pred, (np.ndarray, list)):
                    raise ValueError(f"Predicted labels for {key} must be array-like.")
                if len(y_true_l[0]) != len(y_pred):
                    raise ValueError(f"Length of true labels and predictions for {key} must be equal.")
                detector_dict[key] = (y_pred, y_true_l[0])

            # Adding Null Detectors (always predicting 0 or 1)
            detector_dict["Null Detector 1"] = ([0] * len(y_true_l[0]), y_true_l[0])
            detector_dict["Null Detector 2"] = ([1] * len(y_true_l[0]), y_true_l[0])

            # Adding Random Detector
            np.random.seed(0)
            random_pred = np.where(np.random.rand(len(y_true_l[0])) >= 0.5, 1, 0).tolist()
            detector_dict["Random Detector"] = (random_pred, y_true_l[0])

            data = []

            # Calculating precision, recall, and average detection delay for each detector
            for key, (pred, true) in detector_dict.items():
                precision = round(precision_score(true, pred, zero_division=0), 3)
                recall = round(recall_score(true, pred, zero_division=0), 3)
                avg_delay = round(EvaluationUtils.get_average_detection_delay(true, pred), 3)
                data.append([key, precision, recall, avg_delay])

            report_df = pd.DataFrame(
                columns=["Detector", "Precision", "Recall", "Average Detection Delay"],
                data=data
            )

            logger.debug("Classification report generated successfully.")
            logger.debug(f"\n{report_df}")

            return report_df

        except Exception as e:
            logger.error(f"Error generating classification report: {e}", exc_info=True)
            raise

    @staticmethod
    def anomaly_score(score: List[float], mu: float, sig: float) -> np.ndarray:
        """
        Calculate anomaly scores based on the CDF of a normal distribution.

        Args:
            score (List[float]): List of scores given by the model.
            mu (float): Mean of the normal distribution.
            sig (float): Standard deviation of the normal distribution.

        Returns:
            np.ndarray: Anomaly scores.
        """
        logger = Logger.get_logger(__name__)
        try:
            # Input validation
            if not isinstance(score, (list, np.ndarray)):
                raise ValueError("score must be a list or array-like")
            if not isinstance(mu, (int, float)):
                raise ValueError("mu must be a numeric value")
            if not isinstance(sig, (int, float)):
                raise ValueError("sig must be a numeric value")

            return 1 - stats.norm.sf(score, mu, sig)

        except Exception as e:
            logger.error(f"Error calculating anomaly score: {e}")
            raise

    @staticmethod
    def q_verdict(x: List[float], mu: float, sig: float, n: float = 0.1) -> np.ndarray:
        """
        Provide a verdict on anomaly based on the CDF of a normal distribution.

        Args:
            x (List[float]): List of scores given by the model.
            mu (float): Mean of the normal distribution.
            sig (float): Standard deviation of the normal distribution.
            n (float): Threshold for anomaly detection (default is 0.1).

        Returns:
            np.ndarray: Array of verdicts (1 for anomaly, 0 for normal).
        """
        logger = Logger.get_logger(__name__)
        try:
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

            anomaly_scores = EvaluationUtils.anomaly_score(x.tolist(), mu, sig)
            verdict = np.where(anomaly_scores >= (1 - n), 1, 0)
            logger.debug(f"Generated verdicts with threshold={n}.")

            return verdict

        except Exception as e:
            logger.error(f"Error generating q_verdict: {e}")
            raise