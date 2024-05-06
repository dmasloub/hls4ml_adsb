import os

FEATURES = ["altitude", "groundspeed", "vertical_rate", "x", "y"]
DIFF_FEATURES = ["altitude", "groundspeed", "x", "y"]
STATE_VECTOR_FEATURES = ["groundspeed", "vertical_rate", "longitude", "latitude", "altitude", "heading_unwrapped"]

MODEL_DIR = "trained_models"
MODEL_STANDARD_DIR = os.path.join(MODEL_DIR, "standard")
MODEL_LSTM_DIR = os.path.join(MODEL_DIR, "lstm")

DATA_DIR = "data"
DATA_TRAIN_DIR = os.path.join(DATA_DIR, "train")
DATA_VALIDATION_DIR = os.path.join(DATA_DIR, "validation")
DATA_TEST_DIR = os.path.join(DATA_DIR, "test")
DATA_TEST_DEPARTING_DIR = os.path.join(DATA_TEST_DIR, "departing")
DATA_TEST_LANDING_DIR = os.path.join(DATA_TEST_DIR, "landing")
DATA_TEST_MANOEUVER_DIR = os.path.join(DATA_TEST_DIR, "manoeuver")
DATA_TEST_NOISE_DIR = os.path.join(DATA_TEST_DIR, "noise")


########################################
#             DIFFERENCING             #
########################################

DIFF_DATA = True

########################################
#        SYSTEM HYPERPARAMETERS        #
########################################

K_LAG = 1
K_ORDER = 1
WINDOW_SIZE_STANDARD_AUTOENCODER = WINDOW_SIZE_LSTM_AUTOENCODER = 60
STANDARD_AUTOENCODER_ENCODING_DIMENSION = 10
LSTM_AUTOENCODER_ENCODING_DIMENSION = 10
STANDARD_Q_THRESHOLD = 10e-4
LSTM_Q_THRESHOLD = 10e-2