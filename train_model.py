import pandas as pd
import numpy as np
from tensorflow.keras.callbacks import TensorBoard

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
from tensorflow.keras.layers import Dense, Input

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


def get_standard_autoencoder_model(input_dim, encoding_dim):
  input_layer = Input(shape=input_dim)
  hidden_layer = Dense(encoding_dim, activation="relu")(input_layer)
  output_layer = Dense(input_dim, activation='relu')(hidden_layer)

  autoencoder = Model(inputs=input_layer, outputs=output_layer)

  return autoencoder


autoencoder_model_feature_pipeline = Pipeline(
    steps=[('normalize', StandardScaler())]
)

X_l = []
y_l = []

for df in tqdm(data_dict["train"]):
  X, y = get_windows_data(df[FEATURES], [0] * df.shape[0], window_size=WINDOW_SIZE_STANDARD_AUTOENCODER, tsfresh=True)
  X_l.append(X)
  y_l.append(y)

assert len(X_l) == len(y_l)

X_train_list = []
y_train = np.array([])

for i in tqdm(range(len(X_l))):
    try:
        features = extract_features(X_l[i], column_id="id", column_sort="time", default_fc_parameters=MinimalFCParameters())
        imputed_features = impute(features)
        X_train_list.append(imputed_features)
        y_train = np.append(y_train, y_l[i])
    except Exception as e:
        print(e)
        continue

if X_train_list:  # Ensure the list is not empty
    X_train = pd.concat(X_train_list, ignore_index=True)
else:
    X_train = pd.DataFrame()

X_train_n = autoencoder_model_feature_pipeline.fit_transform(X_train)

logs = "logs/autoencoder/" + datetime.now().strftime("%Y%m%d-%H%M%S")

tboard_callback = TensorBoard(log_dir = logs, histogram_freq = 1, profile_batch = '500,520')

model = get_standard_autoencoder_model(input_dim=X_train_n.shape[1], encoding_dim=STANDARD_AUTOENCODER_ENCODING_DIMENSION)
model.compile(loss='mean_squared_error', optimizer='adam')

model.summary()

history = model.fit(X_train_n, X_train_n,
                    epochs=50,
                    batch_size=128,
                    shuffle=True,
                    callbacks=[tboard_callback]).history

# save model
model.save(MODEL_STANDARD_DIR)