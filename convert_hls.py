import hls4ml
from tensorflow.keras.models import load_model
from utils.constants import *
from utils.plotting import print_dict

model = load_model(MODEL_STANDARD_DIR)
config = hls4ml.utils.config_from_keras_model(model, granularity='model')
print("-----------------------------------")
print("Configuration")
print_dict(config)
print("-----------------------------------")
hls_model = hls4ml.converters.convert_from_keras_model(
    model, hls_config=config, output_dir='hls_model/hls4ml_prj', part='xc7z020clg400-1'
)
hls4ml.utils.plot_model(hls_model,to_file=MODEL_STANDARD_DIR+'/model_plot.png', show_shapes=True, show_precision=True)
hls_model.compile()
hls_model.build(csim=False)
hls4ml.report.read_vivado_report('hls_model/hls4ml_prj')