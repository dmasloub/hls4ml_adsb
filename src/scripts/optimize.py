# src/scripts/optimize.py

import os
from src.config.config import Config
from src.optimizers.bayesian_optimizer import BayesianOptimizer
from src.utils.logger import Logger
from src.utils.common_utils import CommonUtils


def main():
    # Initialize configuration and logger
    config = Config()
    logger = Logger.get_logger(__name__)

    logger.info("Starting Bayesian Optimization process.")

    try:
        # Set random seeds for reproducibility
        CommonUtils.set_seeds(seed=config.optimization.random_state)

        # Ensure necessary directories exist
        CommonUtils.create_directory(config.paths.logs_dir)
        CommonUtils.create_directory(config.paths.checkpoints_dir)

        # Initialize and run the Bayesian Optimizer
        optimizer = BayesianOptimizer(config)
        optimizer.optimize()

    except Exception as e:
        logger.error(f"An error occurred during optimization: {e}", exc_info=True)
        raise

    logger.info("Bayesian Optimization process completed successfully.")


if __name__ == "__main__":
    main()
