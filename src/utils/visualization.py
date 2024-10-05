# src/utils/visualization.py

import os
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict
import pandas as pd
from tensorflow import keras
from src.config.config import Config
from src.utils.logger import Logger


class Visualizer:
    def __init__(self, config: Config):
        """
        Initializes the Visualizer with the provided configuration.

        Args:
            config (Config): Configuration object containing paths and settings.
        """
        self.config = config
        self.logger = Logger.get_logger(__name__)
        sns.set(style="whitegrid")  # Set seaborn style for all plots

    def plot_training_history(self, history: keras.callbacks.History, save_path: str):
        """
        Plots training and validation loss and metrics over epochs.

        Args:
            history (keras.callbacks.History): History object from model training.
            save_path (str): Path to save the plot image.
        """
        try:
            plt.figure(figsize=(12, 6))

            # Plot Loss
            plt.subplot(1, 2, 1)
            plt.plot(history.history['loss'], label='Training Loss')
            if 'val_loss' in history.history:
                plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.title('Model Loss Over Epochs')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()

            # Plot Metrics (e.g., MAE)
            plt.subplot(1, 2, 2)
            for metric in ['mae', 'val_mae']:
                if metric in history.history:
                    plt.plot(history.history[metric], label=metric.capitalize())
            plt.title('Model Metrics Over Epochs')
            plt.xlabel('Epoch')
            plt.ylabel('Metric')
            plt.legend()

            plt.tight_layout()
            self.save_plot(plt, save_path)
            self.logger.info(f"Training history plot saved to {save_path}.")
        except Exception as e:
            self.logger.error(f"Error plotting training history: {e}")
            raise

    def plot_reconstruction_error(self, original: pd.DataFrame, reconstructed: pd.DataFrame, save_path: str):
        """
        Plots the reconstruction error of the autoencoder.

        Args:
            original (pd.DataFrame): Original input data.
            reconstructed (pd.DataFrame): Reconstructed data from the autoencoder.
            save_path (str): Path to save the plot image.
        """
        try:
            error = original - reconstructed
            mse = (error ** 2).mean(axis=1)

            plt.figure(figsize=(10, 6))
            sns.histplot(mse, bins=50, kde=True)
            plt.title('Reconstruction Error Distribution')
            plt.xlabel('Mean Squared Error')
            plt.ylabel('Frequency')

            self.save_plot(plt, save_path)
            self.logger.info(f"Reconstruction error plot saved to {save_path}.")
        except Exception as e:
            self.logger.error(f"Error plotting reconstruction error: {e}")
            raise

    def plot_optimization_progress(self, optimizer_results: List[Dict], save_path: str):
        """
        Plots the progression of Bayesian Optimization, showing how hyperparameters influence performance.

        Args:
            optimizer_results (List[Dict]): List of dictionaries containing hyperparameters and their corresponding scores.
            save_path (str): Path to save the plot image.
        """
        try:
            df = pd.DataFrame(optimizer_results)

            plt.figure(figsize=(10, 6))
            sns.scatterplot(data=df, x='score', y='resource_usage', hue='accuracy', palette='viridis')
            plt.title('Bayesian Optimization Progress')
            plt.xlabel('Objective Score')
            plt.ylabel('Resource Usage (e.g., LUTs)')
            plt.legend(title='Accuracy')

            self.save_plot(plt, save_path)
            self.logger.info(f"Optimization progress plot saved to {save_path}.")
        except Exception as e:
            self.logger.error(f"Error plotting optimization progress: {e}")
            raise

    def plot_resource_utilization(self, resource_metrics: Dict[str, List[int]], save_path: str):
        """
        Plots FPGA resource utilization metrics.

        Args:
            resource_metrics (Dict[str, List[int]]): Dictionary containing resource names and their usage over different configurations.
            save_path (str): Path to save the plot image.
        """
        try:
            df = pd.DataFrame(resource_metrics)
            plt.figure(figsize=(12, 8))
            sns.boxplot(data=df)
            plt.title('FPGA Resource Utilization')
            plt.ylabel('Usage')
            plt.xticks(rotation=45)

            self.save_plot(plt, save_path)
            self.logger.info(f"Resource utilization plot saved to {save_path}.")
        except Exception as e:
            self.logger.error(f"Error plotting resource utilization: {e}")
            raise

    def save_plot(self, plt_obj, save_path: str):
        """
        Saves the current plot to the specified path.

        Args:
            plt_obj: The matplotlib.pyplot object to save.
            save_path (str): Path to save the plot image.
        """
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt_obj.savefig(save_path, bbox_inches='tight')
            plt_obj.close()
            self.logger.debug(f"Plot saved to {save_path}.")
        except Exception as e:
            self.logger.error(f"Error saving plot to {save_path}: {e}")
            raise

    def show_plot(self, plt_obj):
        """
        Displays the current plot interactively.

        Args:
            plt_obj: The matplotlib.pyplot object to display.
        """
        try:
            plt_obj.show()
        except Exception as e:
            self.logger.error(f"Error displaying plot: {e}")
            raise
