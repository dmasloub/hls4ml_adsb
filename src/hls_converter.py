import os
import pandas as pd
from tqdm import tqdm
from src.utils.preprocessing import filter_outliers, diff_data
from src.config import (
    DATA_TRAIN_DIR, DATA_VALIDATION_DIR, DATA_TEST_NOISE_DIR, DATA_TEST_LANDING_DIR,
    DATA_TEST_DEPARTING_DIR, DATA_TEST_MANOEUVER_DIR, DIFF_DATA, DIFF_FEATURES, K_LAG, K_ORDER, STATE_VECTOR_FEATURES
)

class DataLoader:
    def __init__(self, paths=None):
        if paths is None:
            paths = {
                "train": DATA_TRAIN_DIR,
                "validation": DATA_VALIDATION_DIR,
                "test_noise": DATA_TEST_NOISE_DIR,
                "test_landing": DATA_TEST_LANDING_DIR,
                "test_departing": DATA_TEST_DEPARTING_DIR,
                "test_manoeuver": DATA_TEST_MANOEUVER_DIR
            }
        self.data_dict = {key: [] for key in paths}
        self.paths = paths

    def load_data(self):
        for key, path in self.paths.items():
            files = os.listdir(path)
            
            for file in tqdm(files, desc=f"Loading {key} data"):
                file_path = os.path.join(path, file)
                
                # Check if the path is a file
                if os.path.isfile(file_path):
                    df = pd.read_csv(file_path)
                    
                    if key == "train":
                        df = filter_outliers(df, cols=["longitude", "latitude", "altitude", "groundspeed", "x", "y"], std=5)
                    elif key == "validation":
                        df = filter_outliers(df, cols=["longitude", "latitude", "altitude", "groundspeed", "x", "y"], std=8)

                    if DIFF_DATA:
                        df = diff_data(df, cols=DIFF_FEATURES, lag=K_LAG, order=K_ORDER)
                    
                    self.data_dict[key].append(df)

        return self.data_dict

if __name__ == "__main__":
    data_loader = DataLoader()
    data_dict = data_loader.load_data()
    print("Data loaded successfully.")