# src/data_preparation.py

import os
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import MinimalFCParameters
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (
    MODEL_STANDARD_DIR,
    FEATURES,
    WINDOW_SIZE_STANDARD_AUTOENCODER
)
from src.data_loader import DataLoader
from src.utils.utils import get_windows_data

def prepare_data(custom_paths=None, save_path='preprocessed_data.pkl', load_if_exists=True):
    """
    Preprocess data and save to disk. Load preprocessed data if it exists.

    Parameters:
    - custom_paths (dict): Custom paths for datasets.
    - save_path (str): File path to save/load preprocessed data.
    - load_if_exists (bool): If True, load preprocessed data if file exists.

    Returns:
    - preprocessed_data (dict): Dictionary containing preprocessed data.
    """
    if load_if_exists and os.path.exists(save_path):
        print(f"Loading preprocessed data from {save_path}")
        with open(save_path, 'rb') as f:
            preprocessed_data = pickle.load(f)
        return preprocessed_data

    print("Preprocessed data not found. Starting data preparation...")

    # Load data
    data_loader = DataLoader(paths=custom_paths)
    data_dict = data_loader.load_data()

    # Initialize dictionaries to hold preprocessed data
    preprocessed_data = {}

    # Define datasets to process
    datasets = ['train', 'validation', 'test_noise', 'test_landing', 'test_departing', 'test_manoeuver']

    # Process training data first
    dataset = 'train'
    X_windows_list = []
    y_windows_list = []

    for df in tqdm(data_dict.get(dataset, []), desc=f"Processing {dataset} data"):
        if df.empty:
            continue  # Skip empty DataFrames

        X_window, y_window = get_windows_data(
            df[FEATURES],
            df.get("anomaly", [0] * df.shape[0]),
            window_size=WINDOW_SIZE_STANDARD_AUTOENCODER,
            tsfresh=True
        )

        X_windows_list.append(X_window)
        y_windows_list.append(y_window)

    extracted_features_list = []
    concatenated_labels = []

    for X_window, y_window in tqdm(zip(X_windows_list, y_windows_list), desc=f"Extracting features for {dataset}"):
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

    # Save the feature names
    feature_names = X_train.columns.tolist()

    # Preprocess data
    pipeline = Pipeline([('normalize', StandardScaler())])
    X_train_n = pipeline.fit_transform(X_train)

    preprocessed_data[dataset] = {'X': X_train, 'y': y_train, 'X_n': X_train_n}

    # Now process other datasets
    for dataset in datasets:
        if dataset == 'train':
            continue  # already processed
        X_windows_list = []
        y_windows_list = []

        for df in tqdm(data_dict.get(dataset, []), desc=f"Processing {dataset} data"):
            if df.empty:
                continue  # Skip empty DataFrames

            X_window, y_window = get_windows_data(
                df[FEATURES],
                df.get("anomaly", [0] * df.shape[0]),
                window_size=WINDOW_SIZE_STANDARD_AUTOENCODER,
                tsfresh=True
            )

            X_windows_list.append(X_window)
            y_windows_list.append(y_window)

        extracted_features_list = []
        concatenated_labels = []

        for X_window, y_window in tqdm(zip(X_windows_list, y_windows_list), desc=f"Extracting features for {dataset}"):
            features = extract_features(
                X_window,
                column_id="id",
                column_sort="time",
                default_fc_parameters=MinimalFCParameters()
            )
            imputed_features = impute(features)
            extracted_features_list.append(imputed_features)
            concatenated_labels.extend(y_window)

        X = pd.concat(extracted_features_list, ignore_index=True)
        y = np.array(concatenated_labels)

        # Reindex X to have the same columns as training data
        X = X.reindex(columns=feature_names, fill_value=0)

        # Transform using the same pipeline
        X_n = pipeline.transform(X)

        preprocessed_data[dataset] = {'X': X, 'y': y, 'X_n': X_n}

    # Save the pipeline and preprocessed_data to disk
    with open(save_path, 'wb') as f:
        pickle.dump(preprocessed_data, f)

    print(f"Preprocessed data has been saved to {save_path}")

    return preprocessed_data
