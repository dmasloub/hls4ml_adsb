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

def validate_autoencoder(preprocessed_data):
    # Get preprocessed validation data
    X_val_n = preprocessed_data['validation']['X_n']
    y_val = preprocessed_data['validation']['y']

    # Load model
    input_dim = X_val_n.shape[1]
    autoencoder = QuantizedAutoencoder(input_dim=input_dim, encoding_dim=STANDARD_AUTOENCODER_ENCODING_DIMENSION)
    autoencoder.load(MODEL_STANDARD_DIR)

    # Predict
    preds_val = autoencoder.predict(X_val_n)

    # Calculate reconstruction errors
    reconstruction_errors = np.linalg.norm(X_val_n - preds_val, axis=1) ** 2

    # Calculate mean and standard deviation of reconstruction errors
    mu = np.mean(reconstruction_errors)
    std = np.std(reconstruction_errors)

    # Save metrics
    metrics = (mu, std)
    with open(MODEL_STANDARD_DIR + '/metrics.pkl', 'wb') as f:
        pickle.dump(metrics, f)

    return mu, std


if __name__ == "__main__":
    validate_autoencoder()