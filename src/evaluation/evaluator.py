# src/evaluation/evaluator.py

import os
import pickle
from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, accuracy_score
from collections import OrderedDict
from scipy import stats

from src.utils.common_utils import CommonUtils
from src.utils.logger import Logger
from src.utils.evaluation import EvaluationUtils
from src.utils.hls_utils import HLSUtils


class Evaluator:
    def __init__(self, config):
        """
        Initializes the Evaluator with the provided configuration.

        Args:
            config (Config): Configuration object containing paths and settings.
        """
        self.config = config
        self.logger = Logger.get_logger(__name__)

    def evaluate_model(self, model, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Evaluates the autoencoder model on the test dataset.

        Args:
            model (QuantizedAutoencoder): The trained autoencoder model.
            X_test (np.ndarray): Test data features.
            y_test (np.ndarray): Test data labels.

        Returns:
            dict: Dictionary containing evaluation metrics.
        """
        try:
            self.logger.info("Starting model evaluation.")

            # Predict reconstruction
            reconstructed = model.predict(X_test)
            self.logger.debug("Model predictions completed.")

            # Calculate reconstruction errors
            reconstruction_errors = np.linalg.norm(X_test - reconstructed, axis=1) ** 2
            self.logger.debug("Reconstruction errors calculated.")

            # Load mean and std from metrics.pkl
            metrics_path = os.path.join(self.config.paths.model_dir, 'metrics.pkl')
            if not os.path.exists(metrics_path):
                raise FileNotFoundError(f"Metrics file not found at {metrics_path}.")

            with open(metrics_path, 'rb') as f:
                mu, std = pickle.load(f)
            self.logger.debug(f"Loaded metrics: mu={mu}, std={std}")

            # Generate anomaly scores
            anomaly_scores = 1 - stats.norm.sf(reconstruction_errors, mu, std)
            self.logger.debug("Anomaly scores generated.")

            # Generate binary predictions based on a threshold
            threshold = self.config.model.standard_q_threshold
            y_pred = np.where(anomaly_scores >= (1 - threshold), 1, 0)
            self.logger.debug(f"Binary predictions generated with threshold={threshold}.")

            # Calculate accuracy
            accuracy = accuracy_score(y_test, y_pred)
            self.logger.info(f"Model Accuracy: {accuracy}")

            # Calculate precision and recall
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            self.logger.info(f"Model Precision: {precision}")
            self.logger.info(f"Model Recall: {recall}")

            # Calculate average detection delay using EvaluationUtils
            avg_delay = EvaluationUtils.get_average_detection_delay(y_true=y_test.tolist(), y_pred=y_pred.tolist())
            self.logger.info(f"Average Detection Delay: {avg_delay}")

            # Compile metrics
            metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'average_detection_delay': avg_delay,
                'reconstruction_errors': reconstruction_errors,
                'anomaly_scores': anomaly_scores,
                'y_pred': y_pred
            }

            return metrics

        except Exception as e:
            self.logger.error(f"Error during model evaluation: {e}", exc_info=True)
            raise

    def generate_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray, report_path: str) -> pd.DataFrame:
        """
        Generates and saves a classification report.

        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels by the model.
            report_path (str): Path to save the classification report.

        Returns:
            pd.DataFrame: DataFrame containing the classification report.
        """
        try:
            self.logger.info("Generating classification report.")
            # Pass y_true_l as a list containing y_true and provide classifier predictions as kwargs
            report_df = EvaluationUtils.classification_report([y_true.tolist()], Autoencoder=y_pred.tolist())
            report_df.to_csv(report_path, index=False)
            self.logger.info(f"Classification report saved to {report_path}.")
            return report_df
        except Exception as e:
            self.logger.error(f"Error generating classification report: {e}", exc_info=True)
            raise

    def extract_resource_utilization(self, report_path: str) -> Dict[str, Any]:
        """
        Extracts FPGA resource utilization from the HLS synthesis report.

        Args:
            report_path (str): Path to the HLS synthesis report.

        Returns:
            dict: Dictionary containing resource utilization metrics.
        """
        try:
            self.logger.info(f"Extracting resource utilization from report: {report_path}")
            utilization = HLSUtils.extract_utilization(report_path)
            self.logger.info("Resource utilization extracted successfully.")
            self.logger.debug(f"Resource Utilization: {utilization}")
            return utilization
        except Exception as e:
            self.logger.error(f"Error extracting resource utilization: {e}", exc_info=True)
            raise

    def evaluate_resource_utilization(self, utilization: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluates the resource utilization metrics.

        Args:
            utilization (dict): Dictionary containing resource utilization metrics.

        Returns:
            dict: Processed resource utilization metrics.
        """
        try:
            self.logger.info("Evaluating resource utilization metrics.")
            # Example processing: Calculate total resource usage
            total_luts = utilization.get('Total_LUT', 0)
            total_ff = utilization.get('Total_FF', 0)
            total_dsp = utilization.get('Total_DSP48E', 0)

            processed_utilization = {
                'Total_LUT': total_luts,
                'Total_FF': total_ff,
                'Total_DSP48E': total_dsp
            }

            self.logger.info(f"Processed Resource Utilization: {processed_utilization}")
            return processed_utilization

        except Exception as e:
            self.logger.error(f"Error evaluating resource utilization: {e}", exc_info=True)
            raise
