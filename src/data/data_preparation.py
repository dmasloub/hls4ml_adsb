# src/data/data_preparation.py

import pandas as pd
from typing import Dict, List
from src.config.config import Config
from src.utils.logger import Logger


class DataPreparer:
    def __init__(self, config: Config):
        self.config = config
        self.logger = Logger.get_logger()

    def remove_outliers(self, df: pd.DataFrame, cols: List[str], threshold: float) -> pd.DataFrame:
        """
        Remove outliers from specified columns based on the standard deviation threshold.

        Args:
            df (pd.DataFrame): Dataframe to preprocess.
            cols (List[str]): Columns to apply outlier removal.
            threshold (float): Standard deviation threshold.

        Returns:
            pd.DataFrame: Preprocessed dataframe.
        """
        try:
            for col in cols:
                if col not in df.columns:
                    self.logger.warning(
                        f"Column '{col}' not found in dataframe. Skipping outlier removal for this column.")
                    continue
                mean = df[col].mean()
                std = df[col].std()
                initial_count = len(df)
                df = df[(df[col] >= mean - threshold * std) & (df[col] <= mean + threshold * std)]
                final_count = len(df)
                self.logger.info(f"Removed {initial_count - final_count} outliers from column '{col}'.")
            return df
        except Exception as e:
            self.logger.error(f"Error during outlier removal: {e}")
            raise

    def normalize_data(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        """
        Normalize specified columns in the dataframe to a range of [0, 1].

        Args:
            df (pd.DataFrame): Dataframe to preprocess.
            cols (List[str]): Columns to normalize.

        Returns:
            pd.DataFrame: Normalized dataframe.
        """
        try:
            for col in cols:
                if col not in df.columns:
                    self.logger.warning(
                        f"Column '{col}' not found in dataframe. Skipping normalization for this column.")
                    continue
                min_val = df[col].min()
                max_val = df[col].max()
                if max_val - min_val == 0:
                    self.logger.warning(f"Column '{col}' has constant value. Skipping normalization.")
                    continue
                df[col] = (df[col] - min_val) / (max_val - min_val)
                self.logger.info(f"Normalized column '{col}' with min={min_val} and max={max_val}.")
            return df
        except Exception as e:
            self.logger.error(f"Error during normalization: {e}")
            raise

    def difference_data(self, df: pd.DataFrame, cols: List[str], lag: int, order: int) -> pd.DataFrame:
        """
        Apply differencing to specified columns to stabilize the mean.

        Args:
            df (pd.DataFrame): Dataframe to preprocess.
            cols (List[str]): Columns to difference.
            lag (int): Number of lag observations.
            order (int): Number of times differencing is applied.

        Returns:
            pd.DataFrame: Differenced dataframe.
        """
        try:
            for _ in range(order):
                for col in cols:
                    if col not in df.columns:
                        self.logger.warning(
                            f"Column '{col}' not found in dataframe. Skipping differencing for this column.")
                        continue
                    df[col] = df[col].diff(lag)
                    self.logger.info(f"Applied differencing with lag={lag} to column '{col}'.")
            df = df.dropna().reset_index(drop=True)
            self.logger.info("Dropped NaN values after differencing.")
            return df
        except Exception as e:
            self.logger.error(f"Error during differencing: {e}")
            raise

    def prepare_datasets(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Apply all preprocessing steps to each dataset.

        Args:
            datasets (Dict[str, pd.DataFrame]): Dictionary of datasets to preprocess.

        Returns:
            Dict[str, pd.DataFrame]: Dictionary of preprocessed datasets.
        """
        try:
            for dataset_type, df in datasets.items():
                self.logger.info(f"Preprocessing '{dataset_type}' dataset.")

                # Outlier Removal
                if dataset_type == 'train':
                    df = self.remove_outliers(
                        df,
                        self.config.data.features,
                        self.config.data.std_threshold_train
                    )
                elif dataset_type == 'validation':
                    df = self.remove_outliers(
                        df,
                        self.config.data.features,
                        self.config.data.std_threshold_validation
                    )
                else:
                    self.logger.info(f"No outlier removal configured for '{dataset_type}' dataset.")

                # Differencing (if configured and applicable)
                if self.config.data.diff_data and dataset_type.startswith('test'):
                    df = self.difference_data(
                        df,
                        self.config.data.diff_features,
                        lag=self.config.data.k_lag,
                        order=self.config.data.k_order
                    )

                # Normalization
                df = self.normalize_data(df, self.config.data.features)

                # Update the dataset in the dictionary
                datasets[dataset_type] = df
                self.logger.info(f"Completed preprocessing for '{dataset_type}' dataset.")

            return datasets
        except Exception as e:
            self.logger.error(f"Error during dataset preparation: {e}")
            raise
