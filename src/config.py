import os

# Feature Definitions
FEATURES = ["altitude", "groundspeed", "vertical_rate", "x", "y"]
DIFF_FEATURES = ["altitude", "groundspeed", "x", "y"]
STATE_VECTOR_FEATURES = ["groundspeed", "vertical_rate", "longitude", "latitude", "altitude", "heading_unwrapped"]

# Directory Paths
MODEL_DIR = "models"
MODEL_STANDARD_DIR = os.path.join(MODEL_DIR, "standard")

DATA_DIR = "data"
DATA_TRAIN_DIR = os.path.join(DATA_DIR, "train")
DATA_VALIDATION_DIR = os.path.join(DATA_DIR, "validation")
DATA_TEST_DIR = os.path.join(DATA_DIR, "test")
DATA_TEST_DEPARTING_DIR = os.path.join(DATA_TEST_DIR, "departing")
DATA_TEST_LANDING_DIR = os.path.join(DATA_TEST_DIR, "landing")
DATA_TEST_MANOEUVER_DIR = os.path.join(DATA_TEST_DIR, "manoeuver")
DATA_TEST_NOISE_DIR = os.path.join(DATA_TEST_DIR, "noise")

# Differencing Configuration
DIFF_DATA = True

# System Hyperparameters
K_LAG = 1
K_ORDER = 1
WINDOW_SIZE_STANDARD_AUTOENCODER = 30
STANDARD_AUTOENCODER_ENCODING_DIMENSION = 10
STANDARD_Q_THRESHOLD = 10e-4
LEARNING_RATE = 0.001

# Pruning params 
PERCENT = 0.75
BEGIN_STEP = 2000
FREQUENCY = 100