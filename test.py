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

from src.config import (
    MODEL_STANDARD_DIR, FEATURES, WINDOW_SIZE_STANDARD_AUTOENCODER, 
    STANDARD_Q_THRESHOLD, STANDARD_AUTOENCODER_ENCODING_DIMENSION
)
from src.data_loader import DataLoader
from src.model.autoencoder import QuantizedAutoencoder
from src.utils.utils import get_windows_data, q_verdict
from src.utils.evaluation import classification_report
from src.hls_converter import QKerasToHLSConverter

def test_autoencoder(custom_paths=None, build_hls_model=False):
    # Load test data
    data_loader = DataLoader(paths=custom_paths)
    data_dict = data_loader.load_data()
    
    # Load preprocessing pipeline
    with open(MODEL_STANDARD_DIR + '/pipeline.pkl', 'rb') as f:
        pipeline = pickle.load(f)
    
    # Load metrics 
    with open(MODEL_STANDARD_DIR + '/metrics.pkl', 'rb') as f:
        metrics = pickle.load(f)
        
    (mu, std) = metrics

    # Define test datasets
    test_datasets = ['test_noise', 'test_landing', 'test_departing', 'test_manoeuver']
    all_reconstruction_errors_keras = []
    all_reconstruction_errors_hls = []
    actual_labels = {}
    predicted_labels_keras = {}
    predicted_labels_hls = {}

    # Process each test dataset
    for dataset in test_datasets:
        windowed_test_data_list = []
        windowed_test_labels_list = []

        for df in data_dict[dataset]:
            X, y = get_windows_data(df[FEATURES], df["anomaly"], window_size=WINDOW_SIZE_STANDARD_AUTOENCODER, tsfresh=True)
            windowed_test_data_list.append(X)
            windowed_test_labels_list.append(y)

        extracted_test_features_list = []
        concatenated_test_labels = []

        for X_window, y_window in tqdm(zip(windowed_test_data_list, windowed_test_labels_list), desc="Extracting features from test datasets"):
            features = extract_features(
                X_window, 
                column_id="id", 
                column_sort="time", 
                default_fc_parameters=MinimalFCParameters()
            )
            imputed_features = impute(features)
            extracted_test_features_list.append(imputed_features)
            concatenated_test_labels.extend(y_window)

        X_test = pd.concat(extracted_test_features_list, ignore_index=True)
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
        all_reconstruction_errors_keras.append(reconstruction_errors_keras)
        predicted_labels_keras[dataset] = q_verdict(reconstruction_errors_keras, mu, std, STANDARD_Q_THRESHOLD)

        # Convert and test HLS model
        converter = QKerasToHLSConverter(
            model_path=MODEL_STANDARD_DIR,
            output_dir='hls_model/hls4ml_prj',
            build_model=build_hls_model
        )
        converter.convert()

        hls_model = hls4ml.model.hls_model.HLSModel(converter.hls_model.config)
        preds_test_hls = hls_model.predict(X_test_n)
        reconstruction_errors_hls = np.linalg.norm(X_test_n - preds_test_hls, axis=1) ** 2
        all_reconstruction_errors_hls.append(reconstruction_errors_hls)
        predicted_labels_hls[dataset] = q_verdict(reconstruction_errors_hls, mu, std, STANDARD_Q_THRESHOLD)

        # Store actual labels
        actual_labels[dataset] = y_test

    # Calculate and print accuracy scores for each test dataset
    for dataset in test_datasets:
        acc_score_keras = accuracy_score(actual_labels[dataset], predicted_labels_keras[dataset])
        acc_score_hls = accuracy_score(actual_labels[dataset], predicted_labels_hls[dataset])
        print(f"Accuracy score for {dataset} (Keras): {acc_score_keras}")
        print(f"Accuracy score for {dataset} (HLS): {acc_score_hls}")

    # Generate classification report
    classification_report_keras = classification_report(
        [actual_labels[dataset] for dataset in test_datasets], 
        **{dataset: predicted_labels_keras[dataset] for dataset in test_datasets}
    )
    classification_report_hls = classification_report(
        [actual_labels[dataset] for dataset in test_datasets], 
        **{dataset: predicted_labels_hls[dataset] for dataset in test_datasets}
    )
    print("Classification Report (Keras):")
    print(classification_report_keras)
    print("Classification Report (HLS):")
    print(classification_report_hls)

if __name__ == "__main__":
    # Example usage with default paths and building HLS model
    test_autoencoder(build_hls_model=True)

    # Example usage with custom paths and not building HLS model
    # custom_paths = {
    #     "train": "custom_train_path",
    #     "validation": "custom_validation_path",
    #     "test_noise": "custom_test_noise_path",
    #     "test_landing": "custom_test_landing_path",
    #     "test_departing": "custom_test_departing_path",
    #     "test_manoeuver": "custom_test_manoeuver_path"
    # }
    # test_autoencoder(custom_paths=custom_paths, build_hls_model=False)