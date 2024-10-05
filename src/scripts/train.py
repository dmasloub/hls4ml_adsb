# src/scripts/train.py

import os
import pickle
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_schedule
from src.utils.common_utils import CommonUtils
from src.utils.logger import Logger
from src.config.config import Config
from src.models.autoencoder import QuantizedAutoencoder
from src.data.data_loader import DataLoader
from src.data.data_preparation import DataPreparer
from src.utils.visualization import Visualizer
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

def main():
    # Initialize configuration and logger
    config = Config()
    logger = Logger.get_logger(__name__)

    logger.info("Starting training process with pruning.")

    try:
        # Set random seeds for reproducibility
        CommonUtils.set_seeds(seed=config.optimization.random_state)

        # Define the path for preprocessed data
        preprocessed_data_path = config.paths.preprocessed_data_path

        # Load or preprocess data
        if os.path.exists(preprocessed_data_path):
            logger.info(f"Preprocessed data found at '{preprocessed_data_path}'. Loading data.")
            with open(preprocessed_data_path, 'rb') as f:
                preprocessed_datasets = pickle.load(f)
            logger.info("Preprocessed data loaded successfully.")
        else:
            logger.info("Preprocessed data not found. Proceeding with data loading and preprocessing.")
            data_loader = DataLoader(config)
            raw_datasets = data_loader.get_all_datasets()
            logger.info("Raw datasets loaded successfully.")

            data_preparer = DataPreparer(config)
            preprocessed_datasets = data_preparer.prepare_datasets(raw_datasets, save_path=preprocessed_data_path)
            logger.info("Data preprocessing completed and saved.")

        # Access preprocessed training and validation data
        df_train = preprocessed_datasets.get('train')

        if df_train is None or df_train['X_scaled'].size == 0:
            raise ValueError("Training dataset is empty after preprocessing.")

        X_train = df_train['X_scaled']
        y_train = df_train['y']
        logger.info(f"Shape of training data: {X_train.shape}")

        # Define pruning parameters
        pruning_params = {
            'pruning_schedule': pruning_schedule.ConstantSparsity(
                target_sparsity=config.model.pruning_percent,
                begin_step=config.model.begin_step,
                frequency=config.model.frequency
            )
        }

        # Initialize the autoencoder model with pruning
        autoencoder = QuantizedAutoencoder(config.model, X_train.shape[1], pruning_params=pruning_params)
        logger.info("Autoencoder model with pruning initialized.")

        # Define callbacks
        callbacks = [
            EarlyStopping(monitor='loss', patience=10, restore_best_weights=True),
            ModelCheckpoint(
                filepath=os.path.join(config.paths.checkpoints_dir, 'best_model.h5'),
                monitor='loss',
                save_best_only=True,
                save_freq='epoch'
            ),
            TensorBoard(log_dir=os.path.join(config.paths.logs_dir, 'tensorboard_logs'))
        ]

        # Ensure necessary directories exist
        CommonUtils.create_directory(config.paths.checkpoints_dir)
        CommonUtils.create_directory(config.paths.logs_dir)
        CommonUtils.create_directory(config.paths.model_dir)

        # Prepare TensorFlow datasets
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, X_train))
        train_dataset = train_dataset.batch(config.model.batch_size).prefetch(tf.data.AUTOTUNE)
        logger.info("Training TensorFlow dataset prepared.")

        # Train the autoencoder
        history = autoencoder.train(train_data=train_dataset, callbacks=callbacks)

        logger.info("Model training completed.")

        # Strip pruning wrappers
        autoencoder.strip_pruning()
        logger.info("Pruning wrappers stripped from the model.")

        # Plot training history
        visualizer = Visualizer(config)
        plot_path = os.path.join(config.paths.logs_dir, 'training_history.png')
        visualizer.plot_training_history(history, save_path=plot_path)
        logger.info(f"Training history plot saved to {plot_path}.")

        # Save the final model
        model_save_path = os.path.join(config.paths.model_dir, 'autoencoder.h5')
        autoencoder.save_model(model_save_path)
        logger.info(f"Trained model saved to {model_save_path}.")

    except Exception as e:
        logger.error(f"An error occurred during training: {e}", exc_info=True)
        raise

    logger.info("Training process completed successfully.")

if __name__ == "__main__":
    main()
