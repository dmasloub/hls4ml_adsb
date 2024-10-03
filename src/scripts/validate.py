# src/scripts/validate.py

import os
from src.utils.common_utils import CommonUtils
from src.utils.logger import Logger
from src.config.config import get_config
from src.models.autoencoder import QuantizedAutoencoder
from src.data_preparation import prepare_data
from src.evaluation import classification_report
import pickle


def main():
    # Initialize configuration and logger
    config = get_config()
    logger = Logger.get_logger(__name__, log_level=logging.INFO,
                               log_file=os.path.join(config.paths.logs_dir, 'validate.log'))

    logger.info("Starting validation process.")

    try:
        # Load the trained model
        model_path = os.path.join(config.paths.model_standard_dir, 'autoencoder.h5')
        if not os.path.exists(model_path):
            logger.error(f"Trained model not found at {model_path}.")
            raise FileNotFoundError(f"Trained model not found at {model_path}.")

        input_dim = preprocessed_data = prepare_data()['validation']['X_n'].shape[1]
        autoencoder = QuantizedAutoencoder(
            input_dim=input_dim,
            encoding_dim=config.model.encoding_dim,
            bits=config.model.bits,
            integer=config.model.integer_bits,
            alpha=config.model.alpha
        )
        autoencoder.load(model_path)
        logger.info(f"Loaded trained model from {model_path}.")

        # Prepare data
        preprocessed_data = prepare_data()
        X_val_n = preprocessed_data['validation']['X_n']
        y_val = preprocessed_data['validation']['y']

        # Validate the autoencoder
        mu, std = autoencoder.validate(X_val_n)
        logger.info(f"Validation completed with mu={mu} and std={std}.")

        # Save metrics
        metrics_path = os.path.join(config.paths.model_standard_dir, 'metrics.pkl')
        with open(metrics_path, 'wb') as f:
            pickle.dump((mu, std), f)
        logger.info(f"Validation metrics saved to {metrics_path}.")

    except Exception as e:
        logger.error(f"An error occurred during validation: {e}", exc_info=True)
        raise

    logger.info("Validation process completed successfully.")


if __name__ == "__main__":
    main()
