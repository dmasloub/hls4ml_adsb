# src/scripts/test.py

import os
from distutils.command.config import config

from src.config import Config
from src.utils.common_utils import CommonUtils
from src.utils.logger import Logger
from src.config.config import Config
from src.models.autoencoder import QuantizedAutoencoder
from src.data.data_preparation import prepare_data
from src.evaluation.evaluator import Evaluator
from src.converters.hls_converter import HLSConverter


def main():
    # Initialize configuration and logger
    config = Config()
    logger = Logger.get_logger(__name__, log_level=logging.INFO,
                               log_file=os.path.join(config.paths.logs_dir, 'test.log'))

    logger.info("Starting testing process.")

    try:
        # Prepare data
        preprocessed_data = prepare_data()
        logger.info("Data preparation completed.")

        # Load the trained model
        model_path = os.path.join(config.paths.model_standard_dir, 'autoencoder.h5')
        if not os.path.exists(model_path):
            logger.error(f"Trained model not found at {model_path}.")
            raise FileNotFoundError(f"Trained model not found at {model_path}.")

        input_dim = preprocessed_data['test_noise']['X_n'].shape[1]
        autoencoder = QuantizedAutoencoder(
            input_dim=input_dim,
            encoding_dim=config.model.encoding_dim,
            bits=config.model.bits,
            integer=config.model.integer_bits,
            alpha=config.model.alpha
        )
        autoencoder.load(model_path)
        logger.info(f"Loaded trained model from {model_path}.")

        # Initialize Evaluator
        evaluator = Evaluator(config)

        # Evaluate the model on test datasets
        test_datasets = ['test_noise', 'test_landing', 'test_departing', 'test_manoeuver']
        overall_accuracy = {}
        overall_utilization = {}

        for dataset in test_datasets:
            logger.info(f"Evaluating on dataset: {dataset}")
            X_test_n = preprocessed_data[dataset]['X_n']
            y_test = preprocessed_data[dataset]['y']

            # Test the autoencoder (without building HLS model)
            accuracy, utilization = evaluator.evaluate_model(
                model=autoencoder,
                X_test=X_test_n,
                y_test=y_test
            )

            overall_accuracy[dataset] = accuracy
            overall_utilization[dataset] = utilization  # May be None if not built

            # Optionally, you can generate classification reports here
            report_path = os.path.join(config.paths.logs_dir, f'classification_report_{dataset}.csv')
            evaluator.generate_classification_report(
                y_true=y_test,
                y_pred=evaluator.get_predictions(),
                report_path=report_path
            )

        logger.info("Testing on all datasets completed.")
        logger.info(f"Overall Accuracies: {overall_accuracy}")
        logger.info(f"Overall Resource Utilization: {overall_utilization}")

    except Exception as e:
        logger.error(f"An error occurred during testing: {e}", exc_info=True)
        raise

    logger.info("Testing process completed successfully.")


if __name__ == "__main__":
    main()
