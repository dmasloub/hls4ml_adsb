import os
import pickle
import numpy as np
import pandas as pd
import hls4ml
from tensorflow.keras.models import load_model

from src.config import MODEL_STANDARD_DIR, FEATURES, WINDOW_SIZE_STANDARD_AUTOENCODER, STANDARD_AUTOENCODER_ENCODING_DIMENSION
from src.utils.visualization import print_dict

os.environ['LD_PRELOAD'] = '/lib/x86_64-linux-gnu/libudev.so.1'

def convert_to_hls(model_path, output_dir, custom_paths=None):
    # Load the trained model
    model = load_model(model_path)
    
    # Load preprocessing pipeline
    with open(model_path + '/pipeline.pkl', 'rb') as f:
        pipeline = pickle.load(f)
    
    # Create HLS configuration
    config = hls4ml.utils.config_from_keras_model(model, granularity='model')
    
    print("-----------------------------------")
    print("Configuration")
    print_dict(config)
    print("-----------------------------------")
    
    # Convert to HLS model
    hls_model = hls4ml.converters.convert_from_keras_model(
        model, 
        hls_config=config, 
        output_dir=output_dir, 
        backend='VivadoAccelerator',
        board='pynq-z2'  
    )
    
    # Plot the model structure
    hls4ml.utils.plot_model(hls_model, show_shapes=True, show_precision=True, to_file=None)
    
    # Compile the HLS model
    hls_model.compile()
    
    # Print full report
    hls_model.build(csim=False, export=True, bitfile=True)
    
    print("HLS model has been successfully created and compiled.")

if __name__ == "__main__":
    # Convert the trained model to HLS
    output_dir = 'hls_model/hls4ml_prj'
    convert_to_hls(MODEL_STANDARD_DIR, output_dir)