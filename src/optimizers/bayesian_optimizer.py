# src/optimizers/bayesian_optimizer.py

import os
import pickle
from typing import Dict, Any, List
from src.config.config import Config
from src.utils.logger import Logger
from src.utils.common_utils import CommonUtils
from src.models import QuantizedAutoencoder
from src.data.data_loader import DataLoader
from src.data.data_preparation import DataPreparer
from src.evaluation import Evaluator
from src.converters import HLSConverter
from src.utils.visualization import Visualizer
from sklearn.exceptions import NotFittedError
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args


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
        self.best_score = float('inf')
        self.best_hyperparameters = None

        # Define the search space based on the configuration
        self.space = [
            Categorical(self.config.optimization.search_space['bits'], name='bits'),
            Categorical(self.config.optimization.search_space['integer'], name='integer_bits'),
            Real(self.config.optimization.search_space['alpha'][0], self.config.optimization.search_space['alpha'][1],
                 name='alpha'),
            Real(self.config.optimization.search_space['pruning_percent'][0],
                 self.config.optimization.search_space['pruning_percent'][1], name='pruning_percent'),
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

    def _objective(self, bits: int, integer_bits: int, alpha: float, pruning_percent: float, begin_step: int,
                   frequency: int) -> float:
        """
        Objective function to minimize. Combines model accuracy and resource usage.

        Args:
            bits (int): Number of bits for quantization.
            integer_bits (int): Number of integer bits for quantization.
            alpha (float): Scaling factor for quantization.
            pruning_percent (float): Pruning percentage for the model.
            begin_step (int): Step at which pruning begins.
            frequency (int): Frequency of pruning.

        Returns:
            float: Objective score to minimize.
        """
        try:
            self.logger.info(
                f"Evaluating hyperparameters: bits={bits}, integer_bits={integer_bits}, alpha={alpha}, pruning_percent={pruning_percent}, begin_step={begin_step}, frequency={frequency}")

            # Update model configuration
            self.config.model.bits = bits
            self.config.model.integer_bits = integer_bits
            self.config.model.alpha = alpha

            # Initialize components
            data_loader = DataLoader(self.config)
            raw_datasets = data_loader.get_all_datasets()

            data_preparer = DataPreparer(self.config)
            preprocessed_datasets = data_preparer.prepare_datasets(raw_datasets)

            # Initialize and train the model
            autoencoder = QuantizedAutoencoder(self.config.model)
            train_data = preprocessed_datasets['train']
            validation_data = preprocessed_datasets['validation']

            # Convert Pandas DataFrames to TensorFlow Datasets
            train_dataset = tf.data.Dataset.from_tensor_slices((train_data.values, train_data.values))
            train_dataset = train_dataset.batch(self.config.model.batch_size).prefetch(tf.data.AUTOTUNE)

            validation_dataset = tf.data.Dataset.from_tensor_slices((validation_data.values, validation_data.values))
            validation_dataset = validation_dataset.batch(self.config.model.batch_size).prefetch(tf.data.AUTOTUNE)

            # Define callbacks (e.g., pruning callbacks if applicable)
            callbacks = []  # Add any pruning callbacks based on begin_step and frequency if needed

            history = autoencoder.train(
                train_data=train_dataset,
                validation_data=validation_dataset,
                callbacks=callbacks
            )

            # Evaluate the model
            evaluator = Evaluator(self.config)
            evaluation_metrics = evaluator.evaluate(autoencoder, preprocessed_datasets['test_noise'])

            # Extract accuracy (assuming 'accuracy' is part of evaluation_metrics)
            accuracy = evaluation_metrics.get('accuracy', 0)
            self.logger.info(f"Model accuracy: {accuracy}")

            # Convert the model to HLS and extract resource usage
            hls_converter = HLSConverter(self.config)
            hls_output_path = os.path.join(self.config.paths.hls_output_dir,
                                           f"model_bits_{bits}_int_{integer_bits}.cpp")
            hls_converter.convert(autoencoder, save_path=hls_output_path)

            # Extract resource usage metrics
            resource_metrics = hls_converter.get_resource_utilization()
            resource_usage = resource_metrics.get('LUTs', 0)  # Example metric

            self.logger.info(f"Resource usage (LUTs): {resource_usage}")

            # Combine accuracy and resource usage into a single objective score
            # Objective: Minimize (1 - accuracy) * alpha + resource_usage * lambda_reg
            lambda_reg = self.config.optimization.lambda_reg
            objective_score = (1 - accuracy) + lambda_reg * resource_usage

            self.logger.info(f"Objective score: {objective_score}")

            # Append the result
            self.optimizer_results.append({
                'bits': bits,
                'integer_bits': integer_bits,
                'alpha': alpha,
                'pruning_percent': pruning_percent,
                'begin_step': begin_step,
                'frequency': frequency,
                'accuracy': accuracy,
                'resource_usage': resource_usage,
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
                    'begin_step': begin_step,
                    'frequency': frequency
                }
                self.logger.info(f"New best score: {self.best_score} with hyperparameters: {self.best_hyperparameters}")

            # Save checkpoint after each evaluation
            self._save_checkpoint()

            return objective_score

        except Exception as e:
            self.logger.error(f"Error in objective function: {e}")
            # Assign a high penalty score to discourage the optimizer from this region
            return float('inf')

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
            resource_metrics = {}
            for result in self.optimizer_results:
                resource_metrics.setdefault('LUTs', []).append(result['resource_usage'])
            resource_utilization_path = os.path.join(self.config.paths.logs_dir, "resource_utilization.png")
            self.visualizer.plot_resource_utilization(resource_metrics, resource_utilization_path)

        except Exception as e:
            self.logger.error(f"Error during Bayesian Optimization: {e}")
            raise

    def _on_step(self, result):
        """
        Callback function called after each optimization step.

        Args:
            result: The result object from the optimization step.
        """
        self.logger.info(f"Optimization step completed. Current best score: {self.best_score}")

