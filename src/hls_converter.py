import os
import pickle
import hls4ml
import yaml
from tensorflow.keras.models import load_model
from qkeras import QDense, quantized_bits  # Import required components from QKeras
from keras.utils import custom_object_scope
from src.utils.visualization import print_dict

class QKerasToHLSConverter:
    def __init__(self, model_path, output_dir, build_model=False, custom_objects=None):
        self.model_path = model_path
        self.output_dir = output_dir
        self.build_model = build_model
        self.custom_objects = custom_objects if custom_objects else {'QDense': QDense, 'quantized_bits': quantized_bits}
        os.environ['LD_PRELOAD'] = '/lib/x86_64-linux-gnu/libudev.so.1'

    def load_model(self):
        with custom_object_scope(self.custom_objects):
            self.model = load_model(self.model_path)

    def load_pipeline(self):
        with open(self.model_path + '/pipeline.pkl', 'rb') as f:
            self.pipeline = pickle.load(f)

    def create_hls_config(self):
        with open(self.model_path + '/hls4ml_config.yml', 'r') as file:
            self.hls_config = yaml.safe_load(file)

    def convert_to_hls(self):
        self.hls_model = hls4ml.converters.convert_from_keras_model(
            self.model,
            hls_config=self.hls_config,
            output_dir=self.output_dir,
            backend='VivadoAccelerator',
            board='pynq-z2'
        )

    def compile_hls_model(self):
        self.hls_model.compile()
        if self.build_model:
            self.hls_model.build(csim=False, export=True, bitfile=True)

    def plot_model(self):
        hls4ml.utils.plot_model(self.hls_model, show_shapes=True, show_precision=True, to_file=None)

    def convert(self):
        self.load_model()
        self.load_pipeline()
        self.create_hls_config()
        self.convert_to_hls()
        self.plot_model()
        self.compile_hls_model()
        print("HLS model has been successfully created and compiled.")
        print_dict(self.hls_config)

# Example usage
if __name__ == "__main__":
    from src.config import MODEL_STANDARD_DIR  # Ensure this import is correct

    converter = QKerasToHLSConverter(
        model_path=MODEL_STANDARD_DIR,
        output_dir='hls_model/hls4ml_prj',
        build_model=True  # Set to False if you do not want to build the model
    )
    converter.convert()