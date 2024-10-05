# src/models/autoencoder.py

from typing import Optional, List, Union

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from qkeras import QDense, quantized_bits
from src.config.config import ModelConfig
from src.utils.logger import Logger
import tensorflow_model_optimization as tfmot  # Import for pruning

class QuantizedAutoencoder:
    def __init__(self, config: ModelConfig, input_dim, pruning_params=None):
        """
        Initializes the QuantizedAutoencoder with the provided ModelConfig.

        Args:
            config (ModelConfig): Configuration object containing model parameters.
            input_dim (int): Dimension of the input data.
            pruning_params (dict, optional): Parameters for model pruning. Defaults to None.
        """
        self.config = config
        self.input_dim = input_dim
        self.pruning_params = pruning_params  # Store pruning parameters
        self.logger = Logger.get_logger(__name__)  # Use module-specific logger
        self.model = self._build_model()
        self.compile()

    def _build_model(self) -> Model:
        """
        Builds the quantized autoencoder model architecture.

        Returns:
            keras.Model: The constructed autoencoder model.
        """
        try:
            # Accessing configuration parameters directly from ModelConfig
            input_dim = self.input_dim
            encoding_dim = self.config.encoding_dim
            bits = self.config.bits
            integer_bits = self.config.integer_bits
            alpha = self.config.alpha

            # Define the input layer with the correct shape
            input_layer = Input(shape=(input_dim,), name='input_layer')
            self.logger.debug(f"Input layer shape: {(None, input_dim)}")

            # Define the hidden (encoder) layer with quantization
            hidden_layer = QDense(
                units=encoding_dim,
                kernel_quantizer=quantized_bits(bits, integer_bits, alpha=alpha),
                bias_quantizer=quantized_bits(bits, integer_bits, alpha=alpha),
                activation="relu",
                name='hidden_layer'
            )(input_layer)
            self.logger.debug(f"Hidden layer shape: {hidden_layer.shape}")

            # Define the output (decoder) layer with quantization
            output_layer = QDense(
                units=input_dim,
                kernel_quantizer=quantized_bits(bits, integer_bits, alpha=alpha),
                bias_quantizer=quantized_bits(bits, integer_bits, alpha=alpha),
                activation="relu",
                name='output_layer'
            )(hidden_layer)
            self.logger.debug(f"Output layer shape: {output_layer.shape}")

            # Construct the autoencoder model
            autoencoder = Model(inputs=input_layer, outputs=output_layer, name='quantized_autoencoder')

            # Apply pruning if pruning_params is provided
            if self.pruning_params is not None:
                autoencoder = tfmot.sparsity.keras.prune_low_magnitude(autoencoder, **self.pruning_params)
                self.logger.info("Pruning applied to the model.")

            self.logger.info("Quantized Autoencoder model built successfully.")
            self.logger.debug(autoencoder.summary(print_fn=lambda x: self.logger.debug(x)))
            return autoencoder
        except Exception as e:
            self.logger.error(f"Error building the Quantized Autoencoder model: {e}")
            raise

    def compile(self):
        """
        Compiles the autoencoder model with specified optimizer, loss, and metrics.
        """
        try:
            learning_rate = self.config.learning_rate
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            self.model.compile(
                optimizer=optimizer,
                loss='mean_squared_error',
                metrics=['mse', 'mae']
            )
            self.logger.info("Quantized Autoencoder model compiled successfully.")
        except Exception as e:
            self.logger.error(f"Error compiling the Quantized Autoencoder model: {e}")
            raise

    def summary(self):
        """
        Prints the model summary.
        """
        try:
            self.model.summary()
        except Exception as e:
            self.logger.error(f"Error printing model summary: {e}")
            raise

    def train(
        self,
        train_data: tf.data.Dataset,
        validation_data: Optional[tf.data.Dataset] = None,
        callbacks: Optional[List[tf.keras.callbacks.Callback]] = None
    ) -> tf.keras.callbacks.History:
        """
        Trains the autoencoder model on the provided training data.

        Args:
            train_data (tf.data.Dataset): Training dataset.
            validation_data (Optional[tf.data.Dataset], optional): Validation dataset. Defaults to None.
            callbacks (Optional[List[tf.keras.callbacks.Callback]], optional): List of callbacks. Defaults to None.

        Returns:
            tf.keras.callbacks.History: Training history.
        """
        try:
            epochs = self.config.epochs
            self.logger.info("Starting training of the Quantized Autoencoder model.")

            # Ensure pruning callbacks are included if pruning is enabled
            if self.pruning_params is not None:
                if callbacks is None:
                    callbacks = []
                # Check if UpdatePruningStep callback is in callbacks, if not, add it
                if not any(isinstance(cb, tfmot.sparsity.keras.UpdatePruningStep) for cb in callbacks):
                    callbacks.append(tfmot.sparsity.keras.UpdatePruningStep())

            history = self.model.fit(
                train_data,
                epochs=epochs,
                validation_data=validation_data,
                callbacks=callbacks
            )
            self.logger.info("Training completed successfully.")
            return history
        except Exception as e:
            self.logger.error(f"Error during training: {e}")
            raise

    def strip_pruning(self):
        """
        Strips the pruning wrappers from the model after training.
        """
        try:
            self.model = tfmot.sparsity.keras.strip_pruning(self.model)
            self.logger.info("Pruning wrappers stripped from the model.")
        except Exception as e:
            self.logger.error(f"Error stripping pruning wrappers: {e}")
            raise

    def save_model(self, filepath: str):
        """
        Saves the trained model to the specified filepath.

        Args:
            filepath (str): Path to save the model.
        """
        try:
            self.model.save(filepath)
            self.logger.info(f"Model saved successfully at {filepath}.")
        except Exception as e:
            self.logger.error(f"Error saving the model: {e}")
            raise

    def load_model(self, filepath: str):
        """
        Loads a trained model from the specified filepath.

        Args:
            filepath (str): Path from where to load the model.
        """
        try:
            # Define custom_objects dictionary
            custom_objects = {
                'QDense': QDense,
                'quantized_bits': quantized_bits
            }

            # Include pruning custom objects if pruning is enabled
            if self.pruning_params is not None:
                custom_objects.update({
                    'PruneLowMagnitude': tfmot.sparsity.keras.prune_low_magnitude.PruneLowMagnitude
                })

            self.model = tf.keras.models.load_model(filepath, custom_objects=custom_objects, compile=False)
            self.compile()  # Re-compile after loading
            self.logger.info(f"Model loaded successfully from {filepath}.")
        except Exception as e:
            self.logger.error(f"Error loading the model: {e}")
            raise

    def predict(self, data: Union[np.ndarray, tf.data.Dataset]) -> np.ndarray:
        """
        Generates predictions using the trained autoencoder model.

        Args:
            data (Union[np.ndarray, tf.data.Dataset]): Input data for prediction.

        Returns:
            np.ndarray: Predicted outputs.
        """
        try:
            predictions = self.model.predict(data)
            self.logger.info("Predictions generated successfully.")
            return predictions
        except Exception as e:
            self.logger.error(f"Error during prediction: {e}")
            raise
