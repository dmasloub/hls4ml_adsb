# src/optimizers/bayesian_optimizer.py

import os
import pickle
from typing import Dict, Any, List

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
from src.utils.visualization import Visualizer
from sklearn.exceptions import NotFittedError
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import tensorflow_model_optimization as tfmot
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_schedule

class BayesianOptimizer:
    def __init__(self, config: Config):
        """
        Initializes the Bayesian Optimizer with the provided configuration.

        Args:
            config (Config): Configuration object containing optimization settings.
        """
        self.config = config
        self.logger = Logger.get_logger(__name__)
        self.visualizer = Visualizer(config)
        self.optimizer_results = []  # To store optimization results
        self.checkpoint_path = os.path.join(config.paths.checkpoints_dir, "optimization_checkpoint.pkl")
        self.best_score = self.config.optimization.penalty_score
        self.best_hyperparameters = None

        # Define the search space based on the configuration
        self.space = [
            Categorical(self.config.optimization.search_space['bits'], name='bits'),
            Categorical(self.config.optimization.search_space['integer'], name='integer_bits'),
            Real(self.config.optimization.search_space['alpha'][0], self.config.optimization.search_space['alpha'][1],
                 name='alpha'),
            Real(self.config.optimization.search_space['pruning_percent'][0],
                 self.config.optimization.search_space['pruning_percent'][1], name='pruning_percent'),
            Real(self.config.optimization.search_space['standard_q_threshold'][0],
                 self.config.optimization.search_space['standard_q_threshold'][1], name='standard_q_threshold'),
            Integer(self.config.optimization.search_space['begin_step'][0],
                    self.config.optimization.search_space['begin_step'][1], name='begin_step'),
            Integer(self.config.optimization.search_space['frequency'][0],
                    self.config.optimization.search_space['frequency'][1], name='frequency')
        ]

        # Load checkpoint if exists
        self._load_checkpoint()

    def _load_checkpoint(self):
        """
        Loads the optimizer state from a checkpoint file if it exists.
        """
        if os.path.exists(self.checkpoint_path):
            try:
                with open(self.checkpoint_path, 'rb') as f:
                    checkpoint = pickle.load(f)
                    self.optimizer_results = checkpoint['optimizer_results']
                    self.best_score = checkpoint['best_score']
                    self.best_hyperparameters = checkpoint['best_hyperparameters']
                self.logger.info("Loaded optimization checkpoint successfully.")
            except Exception as e:
                self.logger.error(f"Failed to load optimization checkpoint: {e}")
        else:
            self.logger.info("No existing optimization checkpoint found. Starting fresh.")

    def _save_checkpoint(self):
        """
        Saves the optimizer state to a checkpoint file.
        """
        try:
            checkpoint = {
                'optimizer_results': self.optimizer_results,
                'best_score': self.best_score,
                'best_hyperparameters': self.best_hyperparameters
            }
            with open(self.checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint, f)
            self.logger.info("Optimization checkpoint saved successfully.")
        except Exception as e:
            self.logger.error(f"Failed to save optimization checkpoint: {e}")

    def _objective(self, bits: int, integer_bits: int, alpha: float, pruning_percent: float, standard_q_threshold: float, begin_step: int,
                   frequency: int) -> float:
        try:
            self.logger.info(
                f"Evaluating hyperparameters: bits={bits}, integer_bits={integer_bits}, alpha={alpha}, pruning_percent={pruning_percent}, standard_q_threshold={standard_q_threshold} begin_step={begin_step}, frequency={frequency}")

            # Update model configuration
            self.config.model.bits = bits
            self.config.model.integer_bits = integer_bits
            self.config.model.alpha = alpha
            self.config.model.pruning_percent = pruning_percent
            self.config.model.standard_q_threshold = standard_q_threshold
            self.config.model.begin_step = begin_step
            self.config.model.frequency = frequency

            # Load preprocessed data
            preprocessed_data_path = self.config.paths.preprocessed_data_path
            if os.path.exists(preprocessed_data_path):
                self.logger.info(f"Preprocessed data found at '{preprocessed_data_path}'. Loading data.")
                with open(preprocessed_data_path, 'rb') as f:
                    preprocessed_datasets = pickle.load(f)
                self.logger.info("Preprocessed data loaded successfully.")
            else:
                self.logger.info("Preprocessed data not found. Proceeding with data loading and preprocessing.")
                data_loader = DataLoader(self.config)
                raw_datasets = data_loader.get_all_datasets()
                data_preparer = DataPreparer(self.config)
                preprocessed_datasets = data_preparer.prepare_datasets(raw_datasets, save_path=preprocessed_data_path)
                self.logger.info("Data preprocessing completed and saved.")

            # Access preprocessed data
            df_train = preprocessed_datasets['train']

            if df_train is None or df_train['X_scaled'].size == 0:
                raise ValueError("Training dataset is empty after preprocessing.")

            X_train = df_train['X_scaled']
            y_train = df_train['y']
            self.logger.info(f"Shape of training data: {X_train.shape}")

            # Define pruning parameters
            pruning_params = {
                'pruning_schedule': pruning_schedule.ConstantSparsity(
                    target_sparsity=pruning_percent,
                    begin_step=begin_step,
                    frequency=frequency
                )
            }

            # Initialize the autoencoder model with pruning
            autoencoder = QuantizedAutoencoder(self.config.model, X_train.shape[1], pruning_params=pruning_params)
            self.logger.info("Autoencoder model with pruning initialized.")

            # Define callbacks
            callbacks = [
                EarlyStopping(monitor='loss', patience=10, restore_best_weights=True),
                ModelCheckpoint(
                    filepath=os.path.join(self.config.paths.checkpoints_dir, 'best_model.h5'),
                    monitor='loss',
                    save_best_only=True,
                    save_freq='epoch'
                ),
            ]

            # Ensure necessary directories exist
            CommonUtils.create_directory(self.config.paths.checkpoints_dir)
            CommonUtils.create_directory(self.config.paths.logs_dir)
            CommonUtils.create_directory(self.config.paths.model_dir)

            # Prepare TensorFlow datasets
            train_dataset = tf.data.Dataset.from_tensor_slices((X_train, X_train))
            train_dataset = train_dataset.batch(self.config.model.batch_size).prefetch(tf.data.AUTOTUNE)
            self.logger.info("Training TensorFlow dataset prepared.")

            # Train the autoencoder
            history = autoencoder.train(
                train_data=train_dataset,
                callbacks=callbacks
            )
            self.logger.info("Model training completed.")

            # Strip pruning wrappers
            autoencoder.strip_pruning()
            self.logger.info("Pruning wrappers stripped from the model.")

            # Get validation data
            df_validation = preprocessed_datasets.get('validation')
            if df_validation is None:
                raise ValueError("Validation dataset not found in preprocessed datasets.")

            X_validation = df_validation['X_scaled']
            y_validation = df_validation['y']
            self.logger.info(f"Shape of validation data: {X_validation.shape}")
            self.logger.info(f"Shape of validation labels: {y_validation.shape}")

            # Predict
            preds_val = autoencoder.predict(X_validation)

            # Calculate reconstruction errors
            reconstruction_errors = np.linalg.norm(X_validation - preds_val, axis=1) ** 2

            # Calculate mean and standard deviation of reconstruction errors
            mu = np.mean(reconstruction_errors)
            std = np.std(reconstruction_errors)

            metrics = (mu, std)

            metrics_save_path = os.path.join(self.config.paths.model_dir, 'metrics.pkl')
            CommonUtils.save_object(metrics, metrics_save_path)

            self.logger.info(f"Evaluation Metrics: {metrics}")

            # Evaluate the model
            evaluator = Evaluator(self.config, metrics)
            test_data = preprocessed_datasets['test_manoeuver']
            X_test = test_data['X_scaled']
            y_test = test_data['y']

            evaluation_metrics = evaluator.evaluate_model(autoencoder, X_test, y_test)

            # Extract accuracy
            accuracy = evaluation_metrics.get('accuracy', 0)
            self.logger.info(f"Model accuracy: {accuracy}")

            # Extract precision
            precision = evaluation_metrics.get('precision', 0)
            self.logger.info(f"Model precision: {precision}")

            # Extract recall
            recall = evaluation_metrics.get('recall', 0)
            self.logger.info(f"Model recall: {recall}")

            # Check any Metric under 0.7
            if (accuracy < 0.7) or (precision < 0.7) or (recall < 0.7):
                self.logger.warning("Accuracy or Precision or Recall < 0.7. Assigning worst possible score.")
                return self.config.optimization.penalty_score

            # Compute the average metrics
            average_metrics = (accuracy + precision + recall) / 3.0

            # Ensure directories exist
            CommonUtils.create_directory(self.config.paths.model_dir)
            CommonUtils.create_directory(self.config.paths.hls_output_dir)

            # Save the trained model
            temp_model_path = os.path.join(self.config.paths.model_dir, 'temp_autoencoder.h5')
            autoencoder.save_model(temp_model_path)

            # Convert to HLS and extract resource usage
            hls_converter = HLSConverter(build_model=True)
            utilization = hls_converter.convert(model_filename='temp_autoencoder.h5', pipeline_filename='scaling_pipeline.pkl')

            # Extract resource utilization percentages
            lut_utilization_pct = utilization.get('LUT', {}).get('Total', 0) / utilization.get('LUT', {}).get('Available', 0) * 100
            dsp_utilization_pct = utilization.get('DSP48E', {}).get('Total', 0) / utilization.get('DSP48E', {}).get('Available', 0) * 100
            ff_utilization_pct = utilization.get('FF', {}).get('Total', 0) / utilization.get('FF', {}).get('Available', 0) * 100

            # Log resource utilizations
            self.logger.info("Resource utilization percentages:")
            self.logger.info(f"  LUT Utilization (%): {lut_utilization_pct}")
            self.logger.info(f"  DSP48E Utilization (%): {dsp_utilization_pct}")
            self.logger.info(f"  FF Utilization (%): {ff_utilization_pct}")

            # Check if any utilization is outside (0%, 100%)
            if not (0 <= lut_utilization_pct < 100) or not (0 <= dsp_utilization_pct < 100) or not (0 <= ff_utilization_pct < 100):
                self.logger.warning("Resource utilization out of bounds (0% - 100%). Assigning worst possible score.")
                return self.config.optimization.penalty_score

            # Normalize the utilization percentages to [0, 1]
            lut_normalized = lut_utilization_pct / 100.0
            dsp_normalized = dsp_utilization_pct / 100.0
            ff_normalized = ff_utilization_pct / 100.0

            # Compute the average normalized resource usage
            average_normalized_resource_usage = (lut_normalized + dsp_normalized + ff_normalized) / 3.0

            # Compute the objective score
            lambda_reg = self.config.optimization.lambda_reg
            objective_score = (1 - average_metrics) + lambda_reg * average_normalized_resource_usage

            self.logger.info(f"Objective score: {objective_score}")

            # Append the result
            self.optimizer_results.append({
                'bits': bits,
                'integer_bits': integer_bits,
                'alpha': alpha,
                'pruning_percent': pruning_percent,
                'standard_q_threshold': standard_q_threshold,
                'begin_step': begin_step,
                'frequency': frequency,
                'accuracy': accuracy,
                'lut_utilization_pct': lut_utilization_pct,
                'dsp_utilization_pct': dsp_utilization_pct,
                'ff_utilization_pct': ff_utilization_pct,
                'average_normalized_resource_usage': average_normalized_resource_usage,
                'score': objective_score
            })

            # Update the best score and hyperparameters
            if objective_score < self.best_score:
                self.best_score = objective_score
                self.best_hyperparameters = {
                    'bits': bits,
                    'integer_bits': integer_bits,
                    'alpha': alpha,
                    'pruning_percent': pruning_percent,
                    'standard_q_threshold': standard_q_threshold,
                    'begin_step': begin_step,
                    'frequency': frequency
                }
                self.logger.info(f"New best score: {self.best_score} with hyperparameters: {self.best_hyperparameters}")

            # Save checkpoint
            self._save_checkpoint()

            # Clean up temporary files
            if os.path.exists(temp_model_path):
                os.remove(temp_model_path)
                self.logger.debug(f"Temporary model file {temp_model_path} removed.")

            return objective_score

        except Exception as e:
            self.logger.error(f"Error in objective function: {e}", exc_info=True)
            return self.config.optimization.penalty_score

    def optimize(self):
        """
        Runs the Bayesian Optimization process to find the best hyperparameters.
        """
        try:
            self.logger.info("Starting Bayesian Optimization process.")

            @use_named_args(self.space)
            def objective(**params):
                return self._objective(
                    bits=params['bits'],
                    integer_bits=params['integer_bits'],
                    alpha=params['alpha'],
                    pruning_percent=params['pruning_percent'],
                    standard_q_threshold=params['standard_q_threshold'],
                    begin_step=params['begin_step'],
                    frequency=params['frequency']
                )

            # Number of calls already made (from checkpoint)
            n_initial_calls = len(self.optimizer_results)
            remaining_calls = self.config.optimization.total_calls - n_initial_calls
            if remaining_calls <= 0:
                self.logger.info("Total number of optimization calls already reached.")
                return

            self.logger.info(f"Number of optimization calls to perform: {remaining_calls}")

            # Run the optimization
            result = gp_minimize(
                func=objective,
                dimensions=self.space,
                acq_func='EI',  # Expected Improvement
                n_calls=remaining_calls,
                n_initial_points=self.config.optimization.n_initial_points,
                random_state=self.config.optimization.random_state,
                verbose=True,
                callback=[self._on_step]
            )

            self.logger.info("Bayesian Optimization process completed.")
            self.logger.info(f"Best score: {self.best_score} with hyperparameters: {self.best_hyperparameters}")

            # Save all results
            CommonUtils.save_object(self.optimizer_results,
                                    os.path.join(self.config.paths.checkpoints_dir, "optimizer_results.pkl"))

            # Plot optimization progress
            optimization_progress_path = os.path.join(self.config.paths.logs_dir, "optimization_progress.png")
            self.visualizer.plot_optimization_progress(self.optimizer_results, optimization_progress_path)

            # Plot resource utilization if needed
            resource_metrics = {
                'LUT': [res['lut_utilization_pct'] for res in self.optimizer_results],
                'DSP48E': [res['dsp_utilization_pct'] for res in self.optimizer_results],
                'FF': [res['ff_utilization_pct'] for res in self.optimizer_results]
            }
            resource_utilization_path = os.path.join(self.config.paths.logs_dir, "resource_utilization.png")
            self.visualizer.plot_resource_utilization(resource_metrics, resource_utilization_path)

        except Exception as e:
            self.logger.error(f"Error during Bayesian Optimization: {e}", exc_info=True)
            raise

    def _on_step(self, result):
        """
        Callback function called after each optimization step.

        Args:
            result: The result object from the optimization step.
        """
        self.logger.info(f"Optimization step completed. Current best score: {self.best_score}")
