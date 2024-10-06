# train_val_test.py

import os
import pickle
import numpy as np
import tensorflow as tf

from src.config.config import Config
from src.utils.logger import Logger
from src.utils.common_utils import CommonUtils
from src.models.autoencoder import QuantizedAutoencoder
from src.data.data_loader import DataLoader
from src.data.data_preparation import DataPreparer
from src.evaluation.evaluator import Evaluator
from src.converters.hls_converter import HLSConverter
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_schedule


def main():
    # Initialize configuration
    config = Config()

    # Initialize logger
    logger = Logger.get_logger(__name__)

    try:
        logger.info("Starting training and testing process.")

        # Load preprocessed data
        preprocessed_data_path = config.paths.preprocessed_data_path
        if os.path.exists(preprocessed_data_path):
            logger.info(f"Preprocessed data found at '{preprocessed_data_path}'. Loading data.")
            with open(preprocessed_data_path, 'rb') as f:
                preprocessed_datasets = pickle.load(f)
            logger.info("Preprocessed data loaded successfully.")
        else:
            logger.info("Preprocessed data not found. Proceeding with data loading and preprocessing.")
            data_loader = DataLoader(config)
            raw_datasets = data_loader.get_all_datasets()
            data_preparer = DataPreparer(config)
            preprocessed_datasets = data_preparer.prepare_datasets(raw_datasets, save_path=preprocessed_data_path)
            logger.info("Data preprocessing completed and saved.")

        # Access training data
        df_train = preprocessed_datasets.get('train')
        if df_train is None or df_train['X_scaled'].size == 0:
            raise ValueError("Training dataset is empty or not found.")

        X_train = df_train['X_scaled']
        y_train = df_train['y']
        logger.info(f"Shape of training data: {X_train.shape}")

        # Access validation data
        df_validation = preprocessed_datasets.get('validation')
        if df_validation is None or df_validation['X_scaled'].size == 0:
            raise ValueError("Validation dataset is empty or not found.")

        X_validation = df_validation['X_scaled']
        y_validation = df_validation['y']
        logger.info(f"Shape of validation data: {X_validation.shape}")

        # Define pruning parameters based on config
        pruning_params = {
            'pruning_schedule': pruning_schedule.ConstantSparsity(
                target_sparsity=config.model.pruning_percent,
                begin_step=config.model.begin_step,
                frequency=config.model.frequency
            )
        }

        # Initialize the Quantized Autoencoder with hyperparameters from config
        autoencoder = QuantizedAutoencoder(
            config.model,
            input_dim=X_train.shape[1],
            pruning_params=pruning_params
        )
        logger.info("QuantizedAutoencoder initialized with provided hyperparameters.")

        # Define callbacks
        callbacks = [
            # EarlyStopping(monitor='loss', patience=10, restore_best_weights=True),
            ModelCheckpoint(
                filepath=os.path.join(config.paths.checkpoints_dir, 'best_model.h5'),
                monitor='loss',
                save_best_only=True,
                save_freq='epoch'
            ),
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
        logger.info("Starting model training.")
        history = autoencoder.train(
            train_data=train_dataset,
            callbacks=callbacks
        )
        logger.info("Model training completed.")

        # Strip pruning wrappers
        autoencoder.strip_pruning()
        logger.info("Pruning wrappers stripped from the model.")

        # Save the trained model
        model_save_path = os.path.join(config.paths.model_dir, 'trained_autoencoder.h5')
        autoencoder.save_model(model_save_path)
        logger.info(f"Trained model saved to {model_save_path}")

        # Calculate and save reconstruction metrics on validation data
        logger.info("Calculating reconstruction errors on validation data.")
        preds_val = autoencoder.predict(X_validation)
        reconstruction_errors = np.linalg.norm(X_validation - preds_val, axis=1) ** 2
        mu = np.mean(reconstruction_errors)
        std = np.std(reconstruction_errors)
        metrics = (mu, std)

        metrics_save_path = os.path.join(config.paths.model_dir, 'validation_metrics.pkl')
        CommonUtils.save_object(metrics, metrics_save_path)
        logger.info(f"Validation Metrics saved to {metrics_save_path}: Mean={mu}, Std={std}")

        # Evaluate the model on test data
        logger.info("Evaluating the model on test data.")
        evaluator = Evaluator(config, metrics)
        test_data = preprocessed_datasets.get('test_manoeuver')
        if test_data is None or test_data['X_scaled'].size == 0:
            raise ValueError("Test dataset is empty or not found.")

        X_test = test_data['X_scaled']
        y_test = test_data['y']

        evaluation_metrics = evaluator.evaluate_model(autoencoder, X_test, y_test)
        accuracy = evaluation_metrics.get('accuracy', 0)
        precision = evaluation_metrics.get('precision', 0)
        recall = evaluation_metrics.get('recall', 0)

        logger.info("Model Evaluation Metrics on Test Data:")
        logger.info(f"  Accuracy: {accuracy}")
        logger.info(f"  Precision: {precision}")
        logger.info(f"  Recall: {recall}")

        # Print metrics to console
        print("Model Evaluation Metrics on Test Data:")
        print(f"  Accuracy: {accuracy}")
        print(f"  Precision: {precision}")
        print(f"  Recall: {recall}")

        # Convert to HLS and extract resource usage
        logger.info("Converting the trained model to HLS and extracting resource usage.")
        hls_converter = HLSConverter(build_model=True)
        # hls_conversion_path = os.path.join(config.paths.model_dir, 'trained_autoencoder.h5')
        scaling_pipeline_path = os.path.join(config.paths.model_dir,
                                             'scaling_pipeline.pkl')  # Ensure this path is correct
        utilization = hls_converter.convert(
            model_filename='trained_autoencoder.h5',
            pipeline_filename=scaling_pipeline_path
        )

        # Extract resource utilization percentages
        lut_utilization_pct = (utilization.get('LUT', {}).get('Total', 0) / utilization.get('LUT', {}).get('Available',
                                                                                                           1)) * 100
        dsp_utilization_pct = (utilization.get('DSP48E', {}).get('Total', 0) / utilization.get('DSP48E', {}).get(
            'Available', 1)) * 100
        ff_utilization_pct = (utilization.get('FF', {}).get('Total', 0) / utilization.get('FF', {}).get('Available',
                                                                                                        1)) * 100

        # Log resource utilizations
        logger.info("Resource Utilization Percentages:")
        logger.info(f"  LUT Utilization (%): {lut_utilization_pct:.2f}")
        logger.info(f"  DSP48E Utilization (%): {dsp_utilization_pct:.2f}")
        logger.info(f"  FF Utilization (%): {ff_utilization_pct:.2f}")

        # Print resource utilizations to console
        print("Resource Utilization Percentages:")
        print(f"  LUT Utilization (%): {lut_utilization_pct:.2f}")
        print(f"  DSP48E Utilization (%): {dsp_utilization_pct:.2f}")
        print(f"  FF Utilization (%): {ff_utilization_pct:.2f}")

        logger.info("Training and testing process completed successfully.")

    except Exception as e:
        logger.error(f"An error occurred during the training and testing process: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()
