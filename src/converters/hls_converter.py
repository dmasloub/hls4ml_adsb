# src/converters/hls_converter.py

import os
import pickle
import hls4ml
import logging
from tensorflow.keras.models import load_model
from qkeras import QDense, quantized_bits  # Import required components from QKeras
from keras.utils import custom_object_scope
from src.utils.logger import Logger
from src.config.config import Config
from src.utils.hls_utils import HLSUtils


class HLSConverter:
    """
    A class to handle the conversion of a trained Quantized Autoencoder model to HLS using hls4ml.
    It also extracts FPGA resource utilization metrics from the HLS synthesis report.
    """

    def __init__(self, build_model=False):
        """
        Initializes the HLSConverter with the provided configuration.

        Args:
            build_model (bool, optional): If True, compiles and builds the HLS model.
        """
        self.config = Config()
        self.logger = Logger.get_logger(__name__, log_filename='hls_conversion.log')
        self.logger.setLevel(logging.INFO)
        self.build_model = build_model
        self.model = None
        self.hls_model = None
        self.hls_config = None
        self.pipeline = None  # If using a preprocessing pipeline
        self.output_dir = self.config.paths.hls_output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def load_model(self, model_filename='autoencoder.h5'):
        """
        Loads the trained Keras model with QKeras layers.

        Args:
            model_filename (str, optional): Filename of the trained model. Defaults to 'autoencoder.h5'.
        """
        model_path = os.path.join(self.config.paths.model_dir, model_filename)
        self.logger.info(f"Loading trained model from {model_path}.")
        if not os.path.exists(model_path):
            self.logger.error(f"Model file not found at {model_path}.")
            raise FileNotFoundError(f"Model file not found at {model_path}.")

        # Load the model with custom objects
        custom_objects = {'QDense': QDense, 'quantized_bits': quantized_bits}
        with custom_object_scope(custom_objects):
            self.model = load_model(model_path)

        self.logger.info("Model loaded successfully.")

    def load_pipeline(self, pipeline_filename='scaling_pipeline.pkl'):
        """
        Loads the preprocessing pipeline if it exists.

        Args:
            pipeline_filename (str, optional): Filename of the preprocessing pipeline. Defaults to 'scaling_pipeline.pkl'.
        """
        pipeline_path = os.path.join(self.config.paths.model_dir, pipeline_filename)
        self.logger.info(f"Loading preprocessing pipeline from {pipeline_path}.")
        if not os.path.exists(pipeline_path):
            self.logger.warning(f"Pipeline file not found at {pipeline_path}. Proceeding without pipeline.")
            return

        with open(pipeline_path, 'rb') as f:
            self.pipeline = pickle.load(f)
        self.logger.info("Preprocessing pipeline loaded successfully.")

    def create_hls_config(self):
        """
        Creates the HLS configuration based on the loaded Keras model.
        """
        self.logger.info("Creating HLS configuration from the Keras model.")
        self.hls_config = hls4ml.utils.config_from_keras_model(
            self.model,
            granularity='models',
        )
        self.logger.info("HLS configuration created successfully.")
        self.logger.debug(f"HLS Configuration:\n{self.hls_config}")

    def convert_to_hls(self, target='pynq-z2', backend='VivadoAccelerator'):
        """
        Converts the loaded Keras model to an HLS model using hls4ml.

        Args:
            target (str, optional): Target FPGA board. Defaults to 'pynq-z2'.
            backend (str, optional): Backend synthesis tool. Defaults to 'VivadoAccelerator'.
        """
        self.logger.info(f"Converting Keras model to HLS for target='{target}' and backend='{backend}'.")
        self.hls_model = hls4ml.converters.convert_from_keras_model(
            self.model,
            hls_config=self.hls_config,
            output_dir=self.output_dir,
            backend=backend,
            board=target,
        )
        self.logger.info("HLS model conversion completed successfully.")

    def compile_hls_model(self):
        """
        Compiles the HLS model. If build_model is True, it also builds the model.
        """
        if self.hls_model is None:
            self.logger.error("HLS model has not been converted yet. Call convert_to_hls() first.")
            raise ValueError("HLS model has not been converted yet.")

        self.logger.info("Compiling the HLS model.")
        self.hls_model.compile()
        self.logger.info("HLS model compiled successfully.")

        if self.build_model:
            self.logger.info("Building the HLS model.")
            self.hls_model.build(csim=False, export=True, bitfile=True)
            self.logger.info("HLS model built successfully.")

    def plot_model(self):
        """
        Plots the HLS model architecture.
        """
        if self.hls_model is None:
            self.logger.error("HLS model has not been converted yet. Call convert_to_hls() first.")
            raise ValueError("HLS model has not been converted yet.")

        self.logger.info("Plotting the HLS model architecture.")
        hls4ml.utils.plot_model(self.hls_model, show_shapes=True, show_precision=True,
                                to_file=os.path.join(self.output_dir, 'hls_model_plot.png'))
        self.logger.info(f"HLS model architecture plot saved to {self.output_dir}/hls_model_plot.png.")

    def extract_resource_utilization(self, report_filename='myproject_csynth.rpt'):
        """
        Extracts FPGA resource utilization metrics from the HLS synthesis report.

        Args:
            report_filename (str, optional): Filename of the synthesis report. Defaults to 'myproject_csynth.rpt'.

        Returns:
            dict: Dictionary containing resource utilization metrics.
        """
        report_path = os.path.join(self.output_dir, 'myproject_prj', 'solution1', 'syn', 'report', report_filename)
        self.logger.info(f"Extracting resource utilization from report: {report_path}.")
        if not os.path.exists(report_path):
            self.logger.error(f"Synthesis report not found at {report_path}. Ensure that the HLS model has been built.")
            raise FileNotFoundError(f"Synthesis report not found at {report_path}.")

        hls_utils = HLSUtils()
        utilization = hls_utils.extract_utilization(report_path)
        self.logger.info("Resource utilization extracted successfully.")
        self.logger.debug(f"Resource Utilization: {utilization}")
        return utilization

    def convert(self, model_filename='autoencoder.h5', pipeline_filename='scaling_pipeline.pkl', target='pynq-z2',
                backend='VivadoAccelerator'):
        """
        Executes the full conversion process from Keras model to HLS.

        Args:
            model_filename (str, optional): Filename of the trained Keras model. Defaults to 'autoencoder.h5'.
            pipeline_filename (str, optional): Filename of the preprocessing pipeline. Defaults to 'scaling_pipeline.pkl'.
            target (str, optional): Target FPGA board. Defaults to 'pynq-z2'.
            backend (str, optional): Backend synthesis tool. Defaults to 'VivadoAccelerator'.

        Returns:
            dict: Resource utilization metrics.
        """
        try:
            self.load_model(model_filename=model_filename)
            self.load_pipeline(pipeline_filename=pipeline_filename)
            self.create_hls_config()
            self.convert_to_hls(target=target, backend=backend)
            self.plot_model()
            self.compile_hls_model()
            utilization = self.extract_resource_utilization()
            self.logger.info("HLS conversion process completed successfully.")
            return utilization
        except Exception as e:
            self.logger.error(f"Error during HLS conversion: {e}", exc_info=True)
            raise


# Example usage
if __name__ == "__main__":
    from src.utils.common_utils import CommonUtils

    converter = HLSConverter(build_model=True)  # Set to False if you do not want to build the model
    utilization_metrics = converter.convert()

    # Optionally, save the utilization metrics
    utilization_path = os.path.join(converter.output_dir, 'resource_utilization.pkl')
    CommonUtils.save_object(utilization_metrics, utilization_path)
    print(f"Resource utilization metrics saved to {utilization_path}.")
