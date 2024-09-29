import os
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow_model_optimization.python.core.sparsity.keras import prune, pruning_callbacks, pruning_schedule
from tensorflow_model_optimization.sparsity.keras import strip_pruning
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import MinimalFCParameters

from src.config import (
    MODEL_STANDARD_DIR, FEATURES, WINDOW_SIZE_STANDARD_AUTOENCODER,
    STANDARD_AUTOENCODER_ENCODING_DIMENSION, PERCENT, BEGIN_STEP, FREQUENCY, LEARNING_RATE
)
from src.data_loader import DataLoader
from src.model.autoencoder import QuantizedAutoencoder
from src.utils.utils import get_windows_data
from src.utils.visualization import plot_training_history

def train_autoencoder(preprocessed_data, bits=4, integer=0, alpha=1, pruning_percent=0.75, begin_step=2000, frequency=100):
    # Get preprocessed training data
    X_train_n = preprocessed_data['train']['X_n']
    y_train = preprocessed_data['train']['y']

    # Define model
    autoencoder = QuantizedAutoencoder(
        input_dim=X_train_n.shape[1],
        encoding_dim=STANDARD_AUTOENCODER_ENCODING_DIMENSION,
        bits=bits,
        integer=integer,
        alpha=alpha,
    )

    pruning_params = {
        "pruning_schedule": pruning_schedule.ConstantSparsity(pruning_percent, begin_step=begin_step, frequency=frequency)
    }
    pruned_model = prune.prune_low_magnitude(autoencoder.model, **pruning_params)

    # Compile model
    autoencoder.compile()
    pruned_model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse', 'mae'])

    # Train model
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    pruning_callback = pruning_callbacks.UpdatePruningStep()

    history = pruned_model.fit(
        X_train_n, X_train_n,
        epochs=200,
        batch_size=128,
        shuffle=True,
        validation_split=0.2,
        callbacks=[early_stopping_callback, pruning_callback]
    ).history

    # Save model
    stripped_model = strip_pruning(pruned_model)
    stripped_model.save(MODEL_STANDARD_DIR)

    print("Model training completed and saved successfully.")

    return history


if __name__ == "__main__":
    # Example usage with default paths
    train_autoencoder()

    # Example usage with custom paths
    # custom_paths = {
    #     "train": "custom_train_path",
    #     "validation": "custom_validation_path",
    #     "test_noise": "custom_test_noise_path",
    #     "test_landing": "custom_test_landing_path",
    #     "test_departing": "custom_test_departing_path",
    #     "test_manoeuver": "custom_test_manoeuver_path"
    # }
    # train_autoencoder(custom_paths=custom_paths)