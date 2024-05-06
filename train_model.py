import pandas as pd
from utils.constants import *
from utils.data_processing import filter_outliers, diff_data

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