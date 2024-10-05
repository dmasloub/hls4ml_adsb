# src/scripts/test.py

import os
import pickle

import tensorflow as tf
from src.utils.common_utils import CommonUtils
from src.utils.logger import Logger
from src.config.config import Config
from src.models.autoencoder import QuantizedAutoencoder
from src.data.data_loader import DataLoader
from src.data.data_preparation import DataPreparer
from src.evaluation.evaluator import Evaluator  # Assuming you have an Evaluator class
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard


def main():
    # Initialize configuration and logger
    config = Config()
    logger = Logger.get_logger(__name__)  # Automatically assigns 'src_scripts_test.log'

    logger.info("Starting testing process.")

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

        # Get test data (assuming multiple test datasets)
        test_datasets = ['test_noise', 'test_landing', 'test_departing', 'test_manoeuver']
        results = {}

        # Define model
        autoencoder = QuantizedAutoencoder(config.model, preprocessed_datasets.get(test_datasets[0])['X_scaled'].shape[1])

        for test_set in test_datasets:
            df_test = preprocessed_datasets.get(test_set)
            if df_test is None:
                logger.warning(f"No data found for {test_set}. Skipping.")
                continue

            X_test = df_test['X_scaled']
            y_test = df_test['y']
            logger.info(f"Shape of {test_set} data: {X_test.shape}")  # Should be (num_samples, 6)

            # Prepare data for testing as tf.data.Dataset
            test_dataset = tf.data.Dataset.from_tensor_slices(X_test).batch(config.model.batch_size).prefetch(tf.data.AUTOTUNE)
            logger.info(f"{test_set} dataset prepared.")

            # Perform evaluation (assuming Evaluator handles the logic)
            evaluator = Evaluator(config)
            metrics = evaluator.evaluate_model(autoencoder, X_test, y_test)
            results[test_set] = metrics
            logger.info(f"Metrics for {test_set}: {metrics}")

        logger.info(f"All test metrics: {results}")

    except Exception as e:
        logger.error(f"An error occurred during testing: {e}", exc_info=True)
        raise

    logger.info("Testing process completed successfully.")


if __name__ == "__main__":
    main()
