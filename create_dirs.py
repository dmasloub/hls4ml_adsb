from utils.constants import *
import os

if not os.path.exists(MODEL_STANDARD_DIR):
    os.makedirs(MODEL_STANDARD_DIR)

if not os.path.exists(MODEL_LSTM_DIR):
    os.makedirs(MODEL_LSTM_DIR)

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)