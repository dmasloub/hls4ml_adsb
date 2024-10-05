# src/utils/common_utils.py

import os
import pickle
from typing import Dict
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import logging
from src.utils.logger import Logger


class CommonUtils:
    @staticmethod
    def split_dataset(
            df: pd.DataFrame,
            train_size: float,
            validation_size: float,
            test_size: float,
            random_state: int = 42
    ) -> Dict[str, pd.DataFrame]:
        """
        Splits a DataFrame into training, validation, and testing sets.

        Args:
            df (pd.DataFrame): The DataFrame to split.
            train_size (float): Proportion of the dataset to include in the training set.
            validation_size (float): Proportion of the dataset to include in the validation set.
            test_size (float): Proportion of the dataset to include in the testing set.
            random_state (int, optional): Random seed for reproducibility. Defaults to 42.

        Returns:
            Dict[str, pd.DataFrame]: A dictionary containing the split datasets.
        """
        logger = Logger.get_logger(__name__)
        try:
            if not np.isclose(train_size + validation_size + test_size, 1.0):
                raise ValueError("Train, validation, and test sizes must sum to 1.0.")

            train_df, temp_df = train_test_split(
                df,
                test_size=(validation_size + test_size),
                random_state=random_state
            )
            validation_ratio = validation_size / (validation_size + test_size)
            validation_df, test_df = train_test_split(
                temp_df,
                test_size=(1 - validation_ratio),
                random_state=random_state
            )

            logger.info(f"Dataset split into train ({len(train_df)} samples), "
                        f"validation ({len(validation_df)} samples), "
                        f"and test ({len(test_df)} samples) sets.")

            return {
                "train": train_df.reset_index(drop=True),
                "validation": validation_df.reset_index(drop=True),
                "test": test_df.reset_index(drop=True)
            }
        except Exception as e:
            logger.error(f"Error during dataset splitting: {e}")
            raise

    @staticmethod
    def create_directory(path: str):
        """
        Creates a directory if it doesn't already exist.

        Args:
            path (str): The directory path to create.
        """
        logger = Logger.get_logger(__name__)
        try:
            os.makedirs(path, exist_ok=True)
            logger.debug(f"Directory ensured at path: {path}")
        except Exception as e:
            logger.error(f"Failed to create directory at {path}: {e}")
            raise

    @staticmethod
    def get_current_timestamp(format: str = "%Y%m%d_%H%M%S") -> str:
        """
        Generates a current timestamp string.

        Args:
            format (str, optional): The format of the timestamp. Defaults to "%Y%m%d_%H%M%S".

        Returns:
            str: The formatted current timestamp.
        """
        from datetime import datetime
        timestamp = datetime.now().strftime(format)
        return timestamp

    @staticmethod
    def validate_parameters(**kwargs):
        """
        Validates that provided parameters meet specified conditions.

        Usage:
            CommonUtils.validate_parameters(
                bits=lambda x: x > 0,
                integer_bits=lambda x: x >= 0
            )

        Args:
            **kwargs: Parameter names and their corresponding validation functions.

        Raises:
            ValueError: If any parameter fails its validation.
        """
        logger = Logger.get_logger(__name__)
        try:
            for param, condition in kwargs.items():
                if not condition():
                    raise ValueError(f"Validation failed for parameter: {param}")
            logger.debug("All parameters validated successfully.")
        except Exception as e:
            logger.error(f"Parameter validation error: {e}")
            raise

    @staticmethod
    def set_seeds(seed: int = 42):
        """
        Sets random seeds for reproducibility across various libraries.

        Args:
            seed (int, optional): The seed value to set. Defaults to 42.
        """
        import random
        import tensorflow as tf
        logger = Logger.get_logger(__name__)
        try:
            random.seed(seed)
            np.random.seed(seed)
            tf.random.set_seed(seed)
            logger.debug(f"Random seeds set to {seed} for reproducibility.")
        except Exception as e:
            logger.error(f"Error setting random seeds: {e}")
            raise

    @staticmethod
    def save_object(obj, filepath: str):
        """
        Serializes and saves a Python object to a file using pickle.

        Args:
            obj: The Python object to save.
            filepath (str): The path to the file where the object will be saved.
        """
        logger = Logger.get_logger(__name__)
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(obj, f)
            logger.info(f"Object saved successfully at {filepath}.")
        except Exception as e:
            logger.error(f"Failed to save object at {filepath}: {e}")
            raise

    @staticmethod
    def load_object(filepath: str):
        """
        Loads and deserializes a Python object from a file using pickle.

        Args:
            filepath (str): The path to the file from which to load the object.

        Returns:
            The deserialized Python object.
        """
        logger = Logger.get_logger(__name__)
        try:
            with open(filepath, 'rb') as f:
                obj = pickle.load(f)
            logger.info(f"Object loaded successfully from {filepath}.")
            return obj
        except Exception as e:
            logger.error(f"Failed to load object from {filepath}: {e}")
            raise
