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
from src.data_preparation import prepare_data
from src.model.autoencoder import QuantizedAutoencoder
from src.utils.utils import get_windows_data, q_verdict
from src.utils.evaluation import classification_report
from src.hls_converter import QKerasToHLSConverter
from src.utils.hls_utils import extract_utilization


def test_autoencoder(preprocessed_data, build_hls_model=False):
    # Load metrics
    with open(MODEL_STANDARD_DIR + '/metrics.pkl', 'rb') as f:
        metrics = pickle.load(f)

    mu, std = metrics

    # Initialize lists to collect actual and predicted labels
    actual_labels = []
    predicted_labels_hls = []

    # Convert HLS model
    converter = QKerasToHLSConverter(
        model_path=MODEL_STANDARD_DIR,
        output_dir='hls_model/hls4ml_prj',
        build_model=build_hls_model
    )
    converter.convert()

    # If build_hls_model is True, compile and build the HLS model
    if build_hls_model:
        converter.compile_hls_model()

    # Load and test HLS model
    hls_model = converter.hls_model

    # Process each test dataset
    test_datasets = ['test_noise', 'test_landing', 'test_departing', 'test_manoeuver']
    for dataset in test_datasets:
        X_test_n = preprocessed_data[dataset]['X_n']
        y_test = preprocessed_data[dataset]['y']

        # Predict with HLS model
        preds_test_hls = hls_model.predict(np.ascontiguousarray(X_test_n))
        reconstruction_errors_hls = np.linalg.norm(X_test_n - preds_test_hls, axis=1) ** 2
        predicted_labels_hls.extend(q_verdict(reconstruction_errors_hls, mu, std, STANDARD_Q_THRESHOLD))

        # Store actual labels
        actual_labels.extend(y_test)

    # Calculate overall accuracy score
    acc_score_hls = accuracy_score(actual_labels, predicted_labels_hls)
    print(f"Overall Accuracy score (HLS): {acc_score_hls}")

    # Initialize utilization
    utilization = None

    # If build_hls_model is True, extract and return resource utilization
    if build_hls_model:
        report_path = os.path.join(converter.output_dir, 'myproject_prj/solution1/syn/report/myproject_csynth.rpt')
        if os.path.exists(report_path):
            utilization = extract_utilization(report_path)
            print("FPGA Resource Utilization:")
            print(utilization)
        else:
            print("HLS synthesis report not found. Resource utilization cannot be extracted.")

    # Return accuracy and utilization
    return acc_score_hls, utilization


if __name__ == "__main__":

    preprocessed_data = prepare_data()
    # Example usage with default paths and building HLS model
    accuracy, resource_utilization = test_autoencoder(preprocessed_data, build_hls_model=True)
    if resource_utilization:
        # Do something with resource_utilization if needed
        pass

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
