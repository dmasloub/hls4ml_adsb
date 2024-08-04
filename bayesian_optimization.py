import os
import pickle
import numpy as np
import pandas as pd
import optuna
import logging
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import MinimalFCParameters
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow_model_optimization.python.core.sparsity.keras import prune, pruning_schedule, pruning_callbacks
from tensorflow_model_optimization.sparsity.keras import strip_pruning

from src.config import (
    MODEL_STANDARD_DIR, FEATURES, WINDOW_SIZE_STANDARD_AUTOENCODER, 
    STANDARD_Q_THRESHOLD, STANDARD_AUTOENCODER_ENCODING_DIMENSION, 
    DATA_TEST_NOISE_DIR, DATA_TEST_LANDING_DIR, DATA_TEST_DEPARTING_DIR, DATA_TEST_MANOEUVER_DIR
)
from src.data_loader import DataLoader
from src.model.autoencoder import QuantizedAutoencoder
from src.utils.utils import get_windows_data, q_verdict
from src.hls_converter import QKerasToHLSConverter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_test_data():
    """Load and preprocess test datasets."""
    data_loader = DataLoader(paths={
        "test_noise": DATA_TEST_NOISE_DIR,
        "test_landing": DATA_TEST_LANDING_DIR,
        "test_departing": DATA_TEST_DEPARTING_DIR,
        "test_manoeuver": DATA_TEST_MANOEUVER_DIR
    })
    data_dict = data_loader.load_data()
    return data_dict

def load_pipeline_and_metrics():
    """Load preprocessing pipeline and saved metrics."""
    with open(os.path.join(MODEL_STANDARD_DIR, 'pipeline.pkl'), 'rb') as f:
        pipeline = pickle.load(f)
    
    with open(os.path.join(MODEL_STANDARD_DIR, 'metrics.pkl'), 'rb') as f:
        metrics = pickle.load(f)

    return pipeline, metrics

def preprocess_data(data_dict, pipeline):
    """Extract and preprocess features from the test data."""
    test_datasets = ['test_noise', 'test_landing', 'test_departing', 'test_manoeuver']
    all_data = {}

    for dataset in test_datasets:
        windowed_test_data_list = []
        windowed_test_labels_list = []

        for df in data_dict[dataset]:
            X, y = get_windows_data(df[FEATURES], df["anomaly"], window_size=WINDOW_SIZE_STANDARD_AUTOENCODER, tsfresh=True)
            windowed_test_data_list.append(X)
            windowed_test_labels_list.append(y)

        extracted_test_features_list = []
        concatenated_test_labels = []

        for X_window, y_window in tqdm(zip(windowed_test_data_list, windowed_test_labels_list), desc=f"Extracting features from {dataset}"):
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
        all_data[dataset] = (X_test_n, y_test)

    return all_data

def evaluate_model(pruned_model, X_test_n, mu, std):
    """Evaluate the pruned model and return accuracy scores."""
    preds_test_keras = pruned_model.predict(X_test_n)
    reconstruction_errors_keras = np.linalg.norm(X_test_n - preds_test_keras, axis=1) ** 2
    predicted_labels_keras = q_verdict(reconstruction_errors_keras, mu, std, STANDARD_Q_THRESHOLD)
    return predicted_labels_keras

def evaluate_resources(hls_model):
    """Evaluate FPGA resource utilization."""
    report = hls_model.build(csim=False, export=False)
    lut = report['EstimatedResources']['LUT']
    dsp = report['EstimatedResources']['DSP']
    ff = report['EstimatedResources']['FF']

    # Normalize resource usage
    normalized_lut = lut / 100000.0  
    normalized_dsp = dsp / 1000.0  
    normalized_ff = ff / 100000.0  

    return normalized_lut, normalized_dsp, normalized_ff

def objective(trial):
    # Hyperparameters to optimize
    pruning_percent = trial.suggest_uniform('pruning_percent', 0.5, 0.9)
    quantization_bits = trial.suggest_int('quantization_bits', 2, 8)
    integer_bits = trial.suggest_int('integer_bits', 0, quantization_bits - 1)
    alpha = trial.suggest_loguniform('alpha', 0.1, 2.0)
    pruning_begin_step = trial.suggest_int('pruning_begin_step', 0, 1000)
    pruning_frequency = trial.suggest_int('pruning_frequency', 50, 200)

    # Load test data, pipeline, and metrics
    data_dict = load_test_data()
    pipeline, metrics = load_pipeline_and_metrics()
    mu, std = metrics

    # Preprocess test data
    all_data = preprocess_data(data_dict, pipeline)
    
    # Define and train the model
    accuracy_scores = []
    for dataset, (X_test_n, y_test) in all_data.items():
        input_dim = X_test_n.shape[1]
        autoencoder = QuantizedAutoencoder(
            input_dim=input_dim, 
            encoding_dim=STANDARD_AUTOENCODER_ENCODING_DIMENSION, 
            bits=quantization_bits,
            integer=integer_bits,
            alpha=alpha
        )
        
        pruning_params = {
            "pruning_schedule": pruning_schedule.ConstantSparsity(pruning_percent, begin_step=pruning_begin_step, frequency=pruning_frequency)
        }
        pruned_model = prune.prune_low_magnitude(autoencoder.model, **pruning_params)

        # Compile model
        pruned_model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse', 'mae'])

        # Train model
        early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        pruning_callback = pruning_callbacks.UpdatePruningStep()

        pruned_model.fit(
            X_test_n, X_test_n,
            epochs=50,
            batch_size=128,
            shuffle=True,
            validation_split=0.2,
            callbacks=[early_stopping_callback, pruning_callback],
            verbose=0
        )

        # Evaluate accuracy
        predicted_labels_keras = evaluate_model(pruned_model, X_test_n, mu, std)
        acc_score_keras = accuracy_score(y_test, predicted_labels_keras)
        accuracy_scores.append(acc_score_keras)

    # Average accuracy across all datasets
    avg_accuracy = np.mean(accuracy_scores)

    # Convert to HLS and evaluate resource utilization
    converter = QKerasToHLSConverter(
        model_path=MODEL_STANDARD_DIR,
        output_dir='hls_model/hls4ml_prj',
        build_model=True
    )
    converter.convert()

    # Evaluate resource utilization
    normalized_lut, normalized_dsp, normalized_ff = evaluate_resources(converter.hls_model)

    # Objective: Maximize accuracy while minimizing resource utilization
    objective_score = avg_accuracy - (normalized_lut + normalized_dsp + normalized_ff)

    return objective_score

if __name__ == "__main__":
    # Run Bayesian Optimization
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)

    # Print best results
    best_trial = study.best_trial
    print(f"Best trial: Pruning Percent: {best_trial.params['pruning_percent']}, Quantization Bits: {best_trial.params['quantization_bits']}")
    print(f"Integer Bits: {best_trial.params['integer_bits']}, Alpha: {best_trial.params['alpha']}")
    print(f"Pruning Begin Step: {best_trial.params['pruning_begin_step']}, Pruning Frequency: {best_trial.params['pruning_frequency']}")
    print(f"Best trial score: {best_trial.value}")
