# src/scripts/train.py

import os
from src.utils.common_utils import CommonUtils
from src.utils.logger import Logger
from src.config.config import Config as config
from src.models.autoencoder import QuantizedAutoencoder
from src.data.data_preparation import prepare_data
from src.utils.visualization import plot_training_history


def main():
    # Initialize configuration and logger
    logger = Logger.get_logger(__name__, log_level=logging.INFO,
                               log_file=os.path.join(config.paths.logs_dir, 'train.log'))

    logger.info("Starting training process.")

    try:
        # Set random seeds for reproducibility
        CommonUtils.set_seeds(seed=config.model.random_state)

        # Prepare data
        preprocessed_data = prepare_data()
        logger.info("Data preparation completed.")

        # Get training data
        X_train_n = preprocessed_data['train']['X_n']
        y_train = preprocessed_data['train']['y']  # Typically unused for autoencoders

        # Define model
        input_dim = X_train_n.shape[1]
        autoencoder = QuantizedAutoencoder(
            input_dim=input_dim,
            encoding_dim=config.model.encoding_dim,
            bits=config.model.bits,
            integer=config.model.integer_bits,
            alpha=config.model.alpha
        )

        # Train the autoencoder
        history = autoencoder.train(
            X_train=X_train_n,
            validation_split=config.model.validation_split,
            batch_size=config.model.batch_size,
            epochs=config.model.epochs,
            pruning_percent=config.model.pruning_percent,
            begin_step=config.model.begin_step,
            frequency=config.model.frequency
        )
        logger.info("Model training completed.")

        # Plot training history
        plot_path = os.path.join(config.paths.logs_dir, 'training_history.png')
        plot_training_history(history, save_path=plot_path)
        logger.info(f"Training history plot saved to {plot_path}.")

        # Save the trained model
        model_save_path = os.path.join(config.paths.model_standard_dir, 'autoencoder.h5')
        autoencoder.save(model_save_path)
        logger.info(f"Trained model saved to {model_save_path}.")

    except Exception as e:
        logger.error(f"An error occurred during training: {e}", exc_info=True)
        raise

    logger.info("Training process completed successfully.")


if __name__ == "__main__":
    main()
