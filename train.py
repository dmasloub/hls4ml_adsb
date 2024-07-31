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

def train_autoencoder(custom_paths=None):
    # Load data
    data_loader = DataLoader(paths=custom_paths)
    data_dict = data_loader.load_data()

    # Process training data
    X_windows_list = []
    y_windows_list = []

    for df in tqdm(data_dict["train"], desc="Processing training data"):
        if df.empty:
            continue  # Skip empty DataFrames
        
        windowed_data, windowed_labels = get_windows_data(
            df[FEATURES], 
            [0] * df.shape[0], 
            window_size=WINDOW_SIZE_STANDARD_AUTOENCODER, 
            tsfresh=True
        )
        
        X_windows_list.append(windowed_data)
        y_windows_list.append(windowed_labels)

    # Extract features and labels
    extracted_features_list = []
    concatenated_labels = []

    for X_window, y_window in tqdm(zip(X_windows_list, y_windows_list), desc="Extacting features"):
        features = extract_features(
            X_window, 
            column_id="id", 
            column_sort="time", 
            default_fc_parameters=MinimalFCParameters()
        )
        imputed_features = impute(features)
        extracted_features_list.append(imputed_features)
        concatenated_labels.extend(y_window)

    X_train = pd.concat(extracted_features_list, ignore_index=True)
    y_train = np.array(concatenated_labels)

    # Preprocess data
    pipeline = Pipeline([('normalize', StandardScaler())])
    X_train_n = pipeline.fit_transform(X_train)

    # Define model
    autoencoder = QuantizedAutoencoder(
        input_dim=X_train_n.shape[1], 
        encoding_dim=STANDARD_AUTOENCODER_ENCODING_DIMENSION,
    )

    pruning_params = {
        "pruning_schedule": pruning_schedule.ConstantSparsity(PERCENT, begin_step=BEGIN_STEP, frequency=FREQUENCY)
    }
    pruned_model = prune.prune_low_magnitude(autoencoder.model, **pruning_params)

    # Compile model
    autoencoder.compile()
    pruned_model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse', 'mae'])

    # Train model
    # logs = "logs/autoencoder/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    # tboard_callback = TensorBoard(log_dir=logs, histogram_freq=1, profile_batch='500,520')
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    pruning_callback = pruning_callbacks.UpdatePruningStep()
    
    # Print the model summary to verify the architecture
    pruned_model.summary()

    history = pruned_model.fit(
        X_train_n, X_train_n,
        epochs=50,
        batch_size=128,
        shuffle=True,
        validation_split=0.2,
        callbacks=[early_stopping_callback, pruning_callback]
    ).history
    
    # plot training
    plot_training_history(history)

    # Save model
    stripped_model = strip_pruning(pruned_model)
    stripped_model.save(MODEL_STANDARD_DIR)
    # Save pipeline
    with open(MODEL_STANDARD_DIR + '/pipeline.pkl', 'wb') as f:
        pickle.dump(pipeline, f)

    print("Model training completed and saved successfully.")

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