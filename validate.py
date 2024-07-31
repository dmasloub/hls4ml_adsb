import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import MinimalFCParameters

from src.config import MODEL_STANDARD_DIR, FEATURES, WINDOW_SIZE_STANDARD_AUTOENCODER, STANDARD_AUTOENCODER_ENCODING_DIMENSION
from src.data_loader import DataLoader
from src.model.autoencoder import QuantizedAutoencoder
from src.utils.utils import get_windows_data
from src.utils.evaluation import classification_report

def validate_autoencoder(custom_paths=None):
    # Load validation data
    data_loader = DataLoader(paths=custom_paths)
    data_dict = data_loader.load_data()
    
    # Load preprocessing pipeline
    with open(MODEL_STANDARD_DIR + '/pipeline.pkl', 'rb') as f:
        pipeline = pickle.load(f)

    # Process validation data
    validation_windows_list = []
    validation_labels_list = []

    for df in tqdm(data_dict["validation"], desc="Processing validation data"):
        if df.empty:
            continue  # Skip empty DataFrames
        
        windowed_data, windowed_labels = get_windows_data(
            df[FEATURES], 
            [0] * df.shape[0], 
            window_size=WINDOW_SIZE_STANDARD_AUTOENCODER, 
            tsfresh=True
        )
        
        validation_windows_list.append(windowed_data)
        validation_labels_list.append(windowed_labels)
    
    extracted_val_features_list = []
    concatenated_val_labels = []

    for X_window, y_window in tqdm(zip(validation_windows_list, validation_labels_list), desc="Extracting validation features"):
        features = extract_features(
            X_window, 
            column_id="id", 
            column_sort="time", 
            default_fc_parameters=MinimalFCParameters()
        )
        imputed_features = impute(features)
        extracted_val_features_list.append(imputed_features)
        concatenated_val_labels.extend(y_window)

    X_val = pd.concat(extracted_val_features_list, ignore_index=True)
    y_val = np.array(concatenated_val_labels)

    # Preprocess data
    X_val_n = pipeline.transform(X_val)

    # Determine input_dim from preprocessed data
    input_dim = X_val_n.shape[1]

    # Load model
    autoencoder = QuantizedAutoencoder(input_dim=input_dim, encoding_dim=STANDARD_AUTOENCODER_ENCODING_DIMENSION)
    autoencoder.load(MODEL_STANDARD_DIR)

    # Predict
    preds_val = autoencoder.predict(X_val_n)

    # Calculate reconstruction errors
    reconstruction_errors = np.linalg.norm(X_val_n - preds_val, axis=1) ** 2
    reconstruction_errors_df = pd.DataFrame({"reconstruction_errors": reconstruction_errors})

    # Calculate mean and standard deviation of reconstruction errors
    mu = np.mean(reconstruction_errors_df["reconstruction_errors"].values)
    std = np.std(reconstruction_errors_df["reconstruction_errors"].values)

    # Print statistics
    print("Reconstruction Error Statistics:")
    print(reconstruction_errors_df.describe())
    
    metrics = (mu, std)
    
    # Save metrics
    with open(MODEL_STANDARD_DIR + '/metrics.pkl', 'wb') as f:
        pickle.dump(metrics, f)
    
    
    return mu, std

if __name__ == "__main__":
    validate_autoencoder()