# src/data/data_loader.py

import os
import pandas as pd
from typing import Dict, List
from src.config.config import Config
from src.utils.logger import Logger


class DataLoader:
    def __init__(self, config: Config):
        self.config = config
        self.logger = Logger.get_logger()

    def load_dataset(self, dataset_type: str) -> pd.DataFrame:
        """
        Load a specific dataset based on the dataset type.

        Args:
            dataset_type (str): Type of dataset to load (e.g., 'train', 'validation', 'test_noise').

        Returns:
            pd.DataFrame: Loaded dataset.
        """
        try:
            # Dynamically get the directory path from config based on dataset_type
            path_attr = f"data_{dataset_type}_dir"
            dataset_path = getattr(self.config.paths, path_attr)

            if not os.path.exists(dataset_path):
                self.logger.error(f"Directory {dataset_path} does not exist.")
                raise FileNotFoundError(f"Directory {dataset_path} does not exist.")

            all_files = [f for f in os.listdir(dataset_path) if os.path.isfile(os.path.join(dataset_path, f))]

            if not all_files:
                self.logger.warning(f"No files found in directory {dataset_path} for dataset type '{dataset_type}'.")
                return pd.DataFrame()  # Return empty DataFrame if no files are found

            df_list = []
            for file in all_files:
                file_path = os.path.join(dataset_path, file)
                try:
                    df = pd.read_csv(file_path)  # Modify if supporting other formats
                    df_list.append(df)
                    self.logger.info(f"Loaded file {file_path} with {len(df)} records.")
                except Exception as e:
                    self.logger.error(f"Failed to load file {file_path}: {e}")

            if df_list:
                combined_df = pd.concat(df_list, ignore_index=True)
                self.logger.info(
                    f"Combined {len(df_list)} files into a single DataFrame with {len(combined_df)} records for '{dataset_type}' dataset.")
                return combined_df
            else:
                self.logger.warning(f"No valid data loaded for dataset type '{dataset_type}'.")
                return pd.DataFrame()

        except Exception as e:
            self.logger.error(f"Failed to load {dataset_type} dataset: {e}")
            raise

    def get_all_datasets(self) -> Dict[str, pd.DataFrame]:
        """
        Load all required datasets.

        Returns:
            Dict[str, pd.DataFrame]: Dictionary containing all datasets.
        """
        datasets = {}
        dataset_types = ['train', 'validation', 'test_noise', 'test_landing', 'test_departing', 'test_manoeuver']
        for dataset_type in dataset_types:
            datasets[dataset_type] = self.load_dataset(dataset_type)
        return datasets
