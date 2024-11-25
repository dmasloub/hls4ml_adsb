# src/data/data_preparation.py

import os
import pickle
from typing import Dict, List, Any, Generator, Tuple

import pandas as pd
import numpy as np
from tqdm import tqdm
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import MinimalFCParameters
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config.config import Config
from src.utils.logger import Logger
from src.utils.evaluation import EvaluationUtils  
from src.utils.common_utils import CommonUtils


class DataPreparer:
    def __init__(self, config: Config):
        """
        Initializes the DataPreparer with the provided configuration.

        Args:
            config (Config): Configuration object containing preprocessing settings.
        """
        self.config = config
        self.logger = Logger.get_logger(__name__)
        self.pipeline = Pipeline([('normalize', StandardScaler())])

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
                        f"Column '{col}' not found in dataframe. Skipping outlier removal for this column."
                    )
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
                        f"Column '{col}' not found in dataframe. Skipping normalization for this column."
                    )
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
                            f"Column '{col}' not found in dataframe. Skipping differencing for this column."
                        )
                        continue
                    df[col] = df[col].diff(lag)
                    self.logger.info(f"Applied differencing with lag={lag} to column '{col}'.")
            df = df.dropna().reset_index(drop=True)
            self.logger.info("Dropped NaN values after differencing.")
            return df
        except Exception as e:
            self.logger.error(f"Error during differencing: {e}")
            raise

    @staticmethod
    def rolled(data: List[float], window_size: int) -> Generator[List[float], None, None]:
        """
        Generator to yield batches of rows from a data list of specified window size.

        Args:
            data (List[float]): Input data from which windows are generated.
            window_size (int): The size of each window.

        Yields:
            List[float]: Subsequent windows of the input data.
        """
        count = 0
        while count <= len(data) - window_size:
            yield data[count: count + window_size]
            count += 1

    @staticmethod
    def max_rolled(data: List[float], window_size: int) -> List[float]:
        """
        Returns the maximum value for each rolling sliding window.

        Args:
            data (List[float]): List of values from which rolling windows are generated.
            window_size (int): The size of each window.

        Returns:
            List[float]: List of maximum values for each rolling window.
        """
        max_values = []
        for window in DataPreparer.rolled(data, window_size):
            max_values.append(max(window))
        return max_values

    def get_windows_data(
        self,
        data_frame: pd.DataFrame,
        labels: List[int],
        window_size: int,
        tsfresh: bool = True
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Prepare data for autoencoder and tsfresh processing.

        Args:
            data_frame (pd.DataFrame): Input data frame containing features.
            labels (List[int]): Corresponding labels for the data.
            window_size (int): The size of each window.
            tsfresh (bool, optional): Indicator whether to prepare dataframe for tsfresh (add 'id' and 'time' columns). Defaults to True.

        Returns:
            Tuple[pd.DataFrame, np.ndarray]: A tuple (X, y) where X is the processed data and y are the corresponding labels.
        """
        try:
            all_windows = []

            # Iterate over windows generated from the input data frame
            for index, window in enumerate(self.rolled(data_frame.values.tolist(), window_size)):
                window_df = pd.DataFrame(window, columns=data_frame.columns)
                if tsfresh:
                    window_df['id'] = [index] * window_df.shape[0]
                    window_df['time'] = list(range(window_df.shape[0]))
                all_windows.append(window_df)

            # Combine all windows into a single data frame or array
            if all_windows:
                X = pd.concat(all_windows, ignore_index=True) if tsfresh else pd.DataFrame(all_windows)
                y = self.max_rolled(labels, window_size)
                self.logger.debug(f"Generated {len(X)} windows for data.")
            else:
                X = pd.DataFrame() if tsfresh else pd.DataFrame([])
                y = np.array([])

            return X, np.array(y)

        except Exception as e:
            self.logger.error(f"Error during windowing data: {e}")
            raise

    def extract_tsfresh_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features using tsfresh from the windowed data.

        Args:
            X (pd.DataFrame): Windowed data with 'id' and 'time' columns.

        Returns:
            pd.DataFrame: Extracted and imputed features.
        """
        try:
            self.logger.info("Starting feature extraction using tsfresh.")
            features = extract_features(
                X,
                column_id="id",
                column_sort="time",
                default_fc_parameters=MinimalFCParameters()
            )
            imputed_features = impute(features)
            self.logger.info("Feature extraction and imputation completed.")
            return imputed_features
        except Exception as e:
            self.logger.error(f"Error during tsfresh feature extraction: {e}")
            raise

    def scale_features(self, X: pd.DataFrame, fit: bool = False) -> np.ndarray:
        """
        Scale features using the predefined pipeline.

        Args:
            X (pd.DataFrame): DataFrame containing features.
            fit (bool, optional): Whether to fit the scaler. Defaults to False.

        Returns:
            np.ndarray: Scaled features.
        """
        try:
            if fit:
                X_scaled = self.pipeline.fit_transform(X)
                self.logger.info("Feature scaling fitted and applied.")
            else:
                X_scaled = self.pipeline.transform(X)
                self.logger.info("Feature scaling applied.")
            return X_scaled
        except Exception as e:
            self.logger.error(f"Error during feature scaling: {e}")
            raise

    def prepare_datasets(
        self,
        datasets: Dict[str, pd.DataFrame],
        save_path: str = 'preprocessed_data.pkl'
    ) -> Dict[str, Dict[str, Any]]:
        """
        Apply all preprocessing steps to each dataset.

        Args:
            datasets (Dict[str, pd.DataFrame]): Dictionary of datasets to preprocess.
            save_path (str, optional): File path to save/load preprocessed data. Defaults to 'preprocessed_data.pkl'.

        Returns:
            Dict[str, Dict[str, Any]]: Dictionary of preprocessed datasets with features and labels.
        """
        try:
            preprocessed_data = {}
            datasets_to_process = ['train', 'validation', 'test_noise', 'test_landing', 'test_departing', 'test_manoeuver']

            # Process training data first to fit the pipeline
            dataset = 'train'
            df = datasets.get(dataset)
            if df is None or df.empty:
                self.logger.warning(f"No data found for dataset type '{dataset}'. Skipping preprocessing.")
            else:
                self.logger.info(f"Preprocessing '{dataset}' dataset.")

                # Outlier Removal
                df = self.remove_outliers(
                    df,
                    cols=self.config.data.features,
                    threshold=self.config.data.std_threshold_train
                )

                # Differencing
                if self.config.data.diff_data:
                    df = self.difference_data(
                        df,
                        cols=self.config.data.diff_features,
                        lag=self.config.data.k_lag,
                        order=self.config.data.k_order
                    )

                # Windowing
                window_size = self.config.data.window_size
                tsfresh = True
                X_windows, y_windows = self.get_windows_data(
                    data_frame=df[self.config.data.features],
                    labels=df.get('anomaly', [0] * len(df)),
                    window_size=window_size,
                    tsfresh=tsfresh
                )

                if tsfresh:
                    extracted_features = self.extract_tsfresh_features(X_windows)
                else:
                    extracted_features = X_windows

                # Scaling
                X_scaled = self.scale_features(extracted_features, fit=True)

                preprocessed_data[dataset] = {
                    'X': extracted_features,
                    'y': y_windows,
                    'X_scaled': X_scaled
                }

                self.logger.info(f"Completed preprocessing for '{dataset}' dataset.")

                # Save feature names for consistency
                feature_names = extracted_features.columns.tolist()

            for dataset in datasets_to_process:
                if dataset == 'train':
                    continue  # Already processed
                df = datasets.get(dataset)
                if df is None or df.empty:
                    self.logger.warning(f"No data found for dataset type '{dataset}'. Skipping preprocessing.")
                    continue

                self.logger.info(f"Preprocessing '{dataset}' dataset.")

                # Outlier Removal
                if dataset == 'validation':
                    df = self.remove_outliers(
                        df,
                        cols=self.config.data.features,
                        threshold=self.config.data.std_threshold_validation
                    )
                else:
                    self.logger.info(f"No outlier removal configured for '{dataset}' dataset.")

                # Differencing 
                if self.config.data.diff_data:
                    df = self.difference_data(
                        df,
                        cols=self.config.data.diff_features,
                        lag=self.config.data.k_lag,
                        order=self.config.data.k_order
                    )

                # Windowing
                X_windows, y_windows = self.get_windows_data(
                    data_frame=df[self.config.data.features],
                    labels=df.get('anomaly', [0] * len(df)),
                    window_size=window_size,
                    tsfresh=tsfresh
                )

                if tsfresh:
                    extracted_features = self.extract_tsfresh_features(X_windows)
                else:
                    extracted_features = X_windows

                # Reindex to have the same columns as training data
                if tsfresh and 'feature_names' in locals():
                    extracted_features = extracted_features.reindex(columns=feature_names, fill_value=0)
                elif not tsfresh and 'feature_names' in locals():
                    extracted_features = extracted_features.copy()
                    extracted_features.columns = feature_names  # Assuming same order

                # Scaling
                X_scaled = self.scale_features(extracted_features, fit=False)

                preprocessed_data[dataset] = {
                    'X': extracted_features,
                    'y': y_windows,
                    'X_scaled': X_scaled
                }

                self.logger.info(f"Completed preprocessing for '{dataset}' dataset.")

            self.save_preprocessed_data(preprocessed_data, save_path)

            return preprocessed_data
        except Exception as e:
            self.logger.error(f"Error preparing dataset: {e}")
            raise


    def save_preprocessed_data(self, preprocessed_data: Dict[str, Dict[str, Any]], save_path: str):
        """
        Save the preprocessed data and pipeline to disk.

        Args:
            preprocessed_data (Dict[str, Dict[str, Any]]): Dictionary of preprocessed datasets.
            save_path (str): File path to save the preprocessed data.
        """
        try:
             with open(save_path, 'wb') as f:
                   pickle.dump(preprocessed_data, f)
             self.logger.info(f"Preprocessed data has been saved to {save_path}.")

             pipeline_save_path = os.path.join(self.config.paths.model_dir, 'scaling_pipeline.pkl')
             CommonUtils.create_directory(self.config.paths.model_dir)
             with open(pipeline_save_path, 'wb') as f:
                 pickle.dump(self.pipeline, f)
             self.logger.info(f"Scaling pipeline saved at {pipeline_save_path}.")

        except Exception as e:
            self.logger.error(f"Error saving preprocessed data and pipeline: {e}")
            raise
