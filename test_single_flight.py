import os
import pickle
import numpy as np
import pandas as pd
import hls4ml
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import MinimalFCParameters
from src.utils.preprocessing import filter_outliers, diff_data

from src.config import (
    MODEL_STANDARD_DIR, FEATURES, WINDOW_SIZE_STANDARD_AUTOENCODER, 
    STANDARD_Q_THRESHOLD, STANDARD_AUTOENCODER_ENCODING_DIMENSION, 
    DIFF_FEATURES, K_LAG, K_ORDER
)
from src.data_loader import DataLoader
from src.model.autoencoder import QuantizedAutoencoder
from src.utils.utils import get_windows_data, q_verdict
from src.utils.evaluation import classification_report
from src.hls_converter import QKerasToHLSConverter

def test_autoencoder_single_csv(csv_path, build_hls_model=False):
    # Load test data from a single CSV file
    df = pd.read_csv(csv_path)
    
    df = diff_data(df, cols=DIFF_FEATURES, lag=K_LAG, order=K_ORDER)

    # Load preprocessing pipeline
    with open(os.path.join(MODEL_STANDARD_DIR, 'pipeline.pkl'), 'rb') as f:
        pipeline = pickle.load(f)
    
    # Load metrics 
    with open(os.path.join(MODEL_STANDARD_DIR, 'metrics.pkl'), 'rb') as f:
        metrics = pickle.load(f)
        
    (mu, std) = metrics

    # Process the test dataset
    windowed_test_data, windowed_test_labels = get_windows_data(
        df[FEATURES], df["anomaly"], window_size=WINDOW_SIZE_STANDARD_AUTOENCODER, tsfresh=True
    )

    # Debug: Check if windowed_test_data is a DataFrame
    if isinstance(windowed_test_data, pd.DataFrame):
        print("Windowed test data is a DataFrame")
    else:
        print("Windowed test data is not a DataFrame")

    # Extracted features storage
    extracted_test_features = []
    concatenated_test_labels = []

    features = extract_features(
        windowed_test_data, 
        column_id="id",
        column_sort="time",
        default_fc_parameters=MinimalFCParameters()
    )
    imputed_features = impute(features)
    extracted_test_features = imputed_features
    concatenated_test_labels = windowed_test_labels
    

    X_test = extracted_test_features
    y_test = np.array(concatenated_test_labels)

    # Preprocess data
    X_test_n = pipeline.transform(X_test)

    # Determine input_dim from preprocessed data
    input_dim = X_test_n.shape[1]

    # Load and test Keras model
    autoencoder = QuantizedAutoencoder(input_dim=input_dim, encoding_dim=STANDARD_AUTOENCODER_ENCODING_DIMENSION)
    autoencoder.load(MODEL_STANDARD_DIR)
    preds_test_keras = autoencoder.predict(X_test_n)
    reconstruction_errors_keras = np.linalg.norm(X_test_n - preds_test_keras, axis=1) ** 2
    predicted_labels_keras = q_verdict(reconstruction_errors_keras, mu, std, STANDARD_Q_THRESHOLD)

    # Convert and test HLS model
    converter = QKerasToHLSConverter(
        model_path=MODEL_STANDARD_DIR,
        output_dir='hls_model/hls4ml_prj',
        build_model=build_hls_model
    )
    converter.convert()

    hls_model = converter.hls_model
    preds_test_hls = hls_model.predict(np.ascontiguousarray(X_test_n))
    reconstruction_errors_hls = np.linalg.norm(X_test_n - preds_test_hls, axis=1) ** 2
    predicted_labels_hls = q_verdict(reconstruction_errors_hls, mu, std, STANDARD_Q_THRESHOLD)

    # Calculate and print accuracy scores
    acc_score_keras = accuracy_score(y_test, predicted_labels_keras)
    acc_score_hls = accuracy_score(y_test, predicted_labels_hls)
    print(f"Accuracy score (Keras): {acc_score_keras}")
    print(f"Accuracy score (HLS): {acc_score_hls}")

    # Generate classification report
    #classification_report_keras = classification_report(y_test, predicted_labels_keras)
    #classification_report_hls = classification_report(y_test, predicted_labels_hls)

    #print("Classification Report (Keras):")
    #print(classification_report_keras)
    #print("Classification Report (HLS):")
    #print(classification_report_hls)

if __name__ == "__main__":
    # Example usage with a specified CSV file and building HLS model
    csv_path = "test_circle.csv"
    test_autoencoder_single_csv(csv_path, build_hls_model=False)
