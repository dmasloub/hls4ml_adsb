# src/scripts/validate.py

import os
import pickle

import numpy as np
import tensorflow as tf
from src.utils.common_utils import CommonUtils
from src.utils.logger import Logger
from src.config.config import Config
from src.models.autoencoder import QuantizedAutoencoder
from src.data.data_loader import DataLoader
from src.data.data_preparation import DataPreparer
from src.evaluation.evaluator import Evaluator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard


def main():
    # Initialize configuration and logger
    config = Config()
    logger = Logger.get_logger(__name__)  # Automatically assigns 'src_scripts_validate.log'

    logger.info("Starting validation process.")

    try:
        # Set random seeds for reproducibility
        CommonUtils.set_seeds(seed=config.optimization.random_state)

        # Define the path for preprocessed data
        preprocessed_data_path = config.paths.preprocessed_data_path

        # Check if preprocessed data already exists
        if os.path.exists(preprocessed_data_path):
            logger.info(f"Preprocessed data found at '{preprocessed_data_path}'. Loading data.")
            with open(preprocessed_data_path, 'rb') as f:
                preprocessed_datasets = pickle.load(f)
            logger.info("Preprocessed data loaded successfully.")
        else:
            logger.info("Preprocessed data not found. Proceeding with data loading and preprocessing.")
            # Load raw datasets
            data_loader = DataLoader(config)
            raw_datasets = data_loader.get_all_datasets()
            logger.info("Raw datasets loaded successfully.")

            # Initialize DataPreparer and preprocess datasets
            data_preparer = DataPreparer(config)
            preprocessed_datasets = data_preparer.prepare_datasets(raw_datasets, save_path=preprocessed_data_path)
            logger.info("Data preprocessing completed and saved.")

        # Get validation data
        df_validation = preprocessed_datasets.get('validation')
        if df_validation is None:
            raise ValueError("Validation dataset not found in preprocessed datasets.")

        X_validation = df_validation['X_scaled']
        y_validation = df_validation['y']
        logger.info(f"Shape of validation data: {X_validation.shape}")  # Should be (num_samples, 6)
        logger.info(f"Shape of validation labels: {y_validation.shape}")  # Should be (num_samples,)

        # Define model
        autoencoder = QuantizedAutoencoder(config.model, X_validation.shape[1])

        model_save_path = os.path.join(config.paths.model_dir, 'autoencoder.h5')
        autoencoder.load_model(model_save_path)

        # Predict
        preds_val = autoencoder.predict(X_validation)

        # Calculate reconstruction errors
        reconstruction_errors = np.linalg.norm(X_validation - preds_val, axis=1) ** 2

        # Calculate mean and standard deviation of reconstruction errors
        mu = np.mean(reconstruction_errors)
        std = np.std(reconstruction_errors)

        metrics = (mu, std)

        metrics_save_path = os.path.join(config.paths.model_dir, 'metrics.pkl')
        CommonUtils.save_object(metrics, metrics_save_path)

        logger.info(f"Evaluation Metrics: {metrics}")

        return metrics


    except Exception as e:
        logger.error(f"An error occurred during validation: {e}", exc_info=True)
        raise

    logger.info("Validation process completed successfully.")


if __name__ == "__main__":
    main()
