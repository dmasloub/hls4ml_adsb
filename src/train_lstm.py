import pandas as pd
import numpy as np
from keras.src.layers import TimeDistributed
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import Sequential

from tqdm import tqdm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tsfresh import extract_features
from tsfresh.feature_extraction import MinimalFCParameters
from tsfresh.utilities.dataframe_functions import impute
from datetime import datetime

from utils.constants import *
from utils.data_processing import filter_outliers, diff_data, get_windows_data
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, LSTM, Reshape

data_dict = {
    "train": [],
    "validation": [],
    "test_noise": [],
    "test_landing": [],
    "test_departing": [],
    "test_manoeuver": [],
}

paths = {
    "train": DATA_TRAIN_DIR,
    "validation": DATA_VALIDATION_DIR,
    "test_noise": DATA_TEST_NOISE_DIR,
    "test_landing": DATA_TEST_LANDING_DIR,
    "test_departing": DATA_TEST_DEPARTING_DIR,
    "test_manoeuver": DATA_TEST_MANOEUVER_DIR
}

for key, path in paths.items():
    files = os.listdir(path)

    for file in files:
        df = pd.read_csv(os.path.join(path, file))
        if key == "train":
            df = filter_outliers(df, cols=["longitude", "latitude", "altitude", "groundspeed", "x", "y"], std=5)
        if key == "validation":
            df = filter_outliers(df, cols=["longitude", "latitude", "altitude", "groundspeed", "x", "y"], std=8)

        if DIFF_DATA:
            df = diff_data(df, cols=DIFF_FEATURES, lag=K_LAG, order=K_ORDER)

        data_dict[key].append(df)


def get_lstm_autoencoder_model(timesteps, input_dim, encoding_dimension):
    model = Sequential()
    model.add(LSTM(encoding_dimension, activation='relu', input_shape=(timesteps, input_dim), return_sequences=True))

    # Decoder part
    model.add(Dense(input_dim))
    model.add(LSTM(input_dim, activation='relu', return_sequences=True))  # Decoder LSTM

    return model

lstm_autoencoder_model_feature_pipeline = Pipeline(
    steps=[('normalize', StandardScaler())]
)

df_train = pd.concat(data_dict["train"], ignore_index=True)

df_train = pd.DataFrame(columns=[FEATURES], data=lstm_autoencoder_model_feature_pipeline.fit_transform(df_train[FEATURES]))

X_train, y_train = get_windows_data(df_train, [0] * df_train.shape[0], window_size=WINDOW_SIZE_LSTM_AUTOENCODER, tsfresh=False)

print("X_train shape:", X_train.shape)

#X_train = np.reshape(X_train, (X_train.shape[0], WINDOW_SIZE_LSTM_AUTOENCODER, 1))

lstm_model = get_lstm_autoencoder_model(WINDOW_SIZE_LSTM_AUTOENCODER, X_train.shape[2], LSTM_AUTOENCODER_ENCODING_DIMENSION)

lstm_model.compile(optimizer='adam', loss='mse')

logs = "logs/lstm/" + datetime.now().strftime("%Y%m%d-%H%M%S")

tboard_callback = TensorBoard(log_dir = logs, histogram_freq = 1, profile_batch = '500,520')

history = lstm_model.fit(X_train, X_train,
                    epochs=50,
                    batch_size=128,
                    shuffle=False,
                    callbacks=[tboard_callback]).history

# save model
lstm_model.save(MODEL_LSTM_DIR)