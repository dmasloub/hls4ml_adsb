# src/models/autoencoder.py

from typing import Optional
import tensorflow as tf
from tensorflow import keras
from qkeras import QDense, QActivation
from qkeras.quantizers import quantized_bits, quantized_relu
from src.config.config import ModelConfig
from src.utils.logger import Logger

class QuantizedAutoencoder:
    def __init__(self, config: ModelConfig):
        """
        Initializes the QuantizedAutoencoder with the provided configuration.

        Args:
            config (ModelConfig): Configuration object containing model parameters.
        """
        self.config = config
        self.logger = Logger.get_logger()
        self.model = self._build_model()
        self.compile_model()

    def _build_model(self) -> keras.Model:
        """
        Builds the quantized autoencoder model architecture.

        Returns:
            keras.Model: The constructed autoencoder model.
        """
        try:
            input_dim = self.config.input_dim
            encoding_dim = self.config.encoding_dim

            input_layer = keras.Input(shape=(input_dim,), name='input_layer')

            # Encoder
            encoder = QDense(
                units=encoding_dim,
                name='encoder_dense',
                kernel_quantizer=quantized_bits(self.config.bits, self.config.integer_bits, alpha=self.config.alpha),
                bias_quantizer=quantized_bits(self.config.bits, self.config.integer_bits, alpha=self.config.alpha)
            )(input_layer)
            encoder = QActivation(
                activation='relu',
                name='encoder_activation'
            )(encoder)

            # Decoder
            decoder = QDense(
                units=input_dim,
                name='decoder_dense',
                kernel_quantizer=quantized_bits(self.config.bits, self.config.integer_bits, alpha=self.config.alpha),
                bias_quantizer=quantized_bits(self.config.bits, self.config.integer_bits, alpha=self.config.alpha)
            )(encoder)
            decoder = QActivation(
                activation='sigmoid',
                name='decoder_activation'
            )(decoder)

            # Autoencoder Model
            autoencoder = keras.Model(inputs=input_layer, outputs=decoder, name='quantized_autoencoder')
            self.logger.info("Quantized Autoencoder model built successfully.")
            return autoencoder
        except Exception as e:
            self.logger.error(f"Error building the Quantized Autoencoder model: {e}")
            raise

    def compile_model(self):
        """
        Compiles the autoencoder model with specified optimizer, loss, and metrics.
        """
        try:
            optimizer = keras.optimizers.Adam(learning_rate=self.config.learning_rate)
            self.model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
            self.logger.info("Quantized Autoencoder model compiled successfully.")
        except Exception as e:
            self.logger.error(f"Error compiling the Quantized Autoencoder model: {e}")
            raise

    def train(
        self,
        train_data: tf.data.Dataset,
        validation_data: Optional[tf.data.Dataset] = None,
        callbacks: Optional[List[keras.callbacks.Callback]] = None
    ) -> keras.callbacks.History:
        """
        Trains the autoencoder model on the provided training data.

        Args:
            train_data (tf.data.Dataset): Training dataset.
            validation_data (Optional[tf.data.Dataset], optional): Validation dataset. Defaults to None.
            callbacks (Optional[List[keras.callbacks.Callback]], optional): List of callbacks. Defaults to None.

        Returns:
            keras.callbacks.History: Training history.
        """
        try:
            self.logger.info("Starting training of the Quantized Autoencoder model.")
            history = self.model.fit(
                train_data,
                epochs=self.config.epochs,
                validation_data=validation_data,
                callbacks=callbacks
            )
            self.logger.info("Training completed successfully.")
            return history
        except Exception as e:
            self.logger.error(f"Error during training: {e}")
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
            self.model = keras.models.load_model(filepath, compile=False)
            self.compile_model()  # Re-compile after loading
            self.logger.info(f"Model loaded successfully from {filepath}.")
        except Exception as e:
            self.logger.error(f"Error loading the model: {e}")
            raise

    def predict(self, data: tf.data.Dataset) -> tf.Tensor:
        """
        Generates predictions using the trained autoencoder model.

        Args:
            data (tf.data.Dataset): Input data for prediction.

        Returns:
            tf.Tensor: Predicted outputs.
        """
        try:
            predictions = self.model.predict(data)
            self.logger.info("Predictions generated successfully.")
            return predictions
        except Exception as e:
            self.logger.error(f"Error during prediction: {e}")
            raise
