import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from qkeras import QDense, quantized_bits

class QuantizedAutoencoder:
    def __init__(self, input_dim, encoding_dim, bits=6, integer=0, alpha=1, learning_rate=0.001):
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.bits = bits
        self.integer = integer
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.model = self._build_model()

    def _build_model(self):
        input_layer = Input(shape=(self.input_dim,))
        
        hidden_layer = QDense(
            self.encoding_dim,
            kernel_quantizer=quantized_bits(self.bits, self.integer, alpha=self.alpha),
            bias_quantizer=quantized_bits(self.bits, self.integer, alpha=self.alpha),
            activation="relu"
        )(input_layer)
        
        output_layer = QDense(
            self.input_dim,
            kernel_quantizer=quantized_bits(self.bits, self.integer, alpha=self.alpha),
            bias_quantizer=quantized_bits(self.bits, self.integer, alpha=self.alpha),
            activation="relu"
        )(hidden_layer)
        
        autoencoder = Model(inputs=input_layer, outputs=output_layer)
        return autoencoder

    def compile(self):
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mse', 'mae'])

    def summary(self):
        return self.model.summary()

    def train(self, X_train, batch_size=128, epochs=50, validation_split=0.2, callbacks=None):
        return self.model.fit(X_train, X_train, batch_size=batch_size, epochs=epochs,
                              validation_split=validation_split, shuffle=True, callbacks=callbacks)

    def save(self, model_dir):
        self.model.save(model_dir)

    def load(self, model_dir):
        self.model = tf.keras.models.load_model(model_dir)

    def predict(self, X):
        return self.model.predict(X)