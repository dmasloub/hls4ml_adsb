# optimization.py
from src.config import MODEL_STANDARD_DIR
from src.data_preparation import prepare_data
from src.utils.hls_utils import extract_utilization
from train import train_autoencoder
from validate import validate_autoencoder
from test import test_autoencoder
from src.hls_converter import QKerasToHLSConverter
from skopt import gp_minimize
from skopt.space import Integer, Real
from skopt.utils import use_named_args

def objective_function(params, lambda_reg=0.5):
    # Extract parameters
    pruning_percent = params['pruning_percent']
    begin_step = params['begin_step']
    frequency = params['frequency']

    # Train the autoencoder with given parameters
    train_autoencoder(
        pruning_percent=pruning_percent,
        begin_step=begin_step,
        frequency=frequency
    )

    # Validate the autoencoder to get reconstruction error statistics
    mu, std = validate_autoencoder()

    # Test the autoencoder to get accuracy and resource utilization
    accuracy, utilization = test_autoencoder(build_hls_model=False)

    # Convert to HLS and get resource utilization
    # Note: Since the HLS model is already built in test_autoencoder, you can skip reconversion

    # Get the LUT utilization (either total or percentage)
    total_lut = utilization.get('Total_LUT', 0)
    # lut_utilization_percent = utilization.get('Utilization (%)_LUT', 0)

    # Combine accuracy and utilization into a single score
    score = accuracy - lambda_reg * total_lut  # Adjust lambda_reg as needed

    return -score  # Negative because most optimizers minimize the objective function



# Define the search space
search_space = [
    Real(0.5, 0.95, name='pruning_percent'),
    Integer(1000, 5000, name='begin_step'),
    Integer(50, 500, name='frequency'),
]

# Prepare the data once
preprocessed_data = prepare_data()

# Now define the objective function wrapper to include preprocessed_data
def objective(**params):
    return objective_function(params, preprocessed_data)

# Run the optimization
res = gp_minimize(
    func=objective,
    dimensions=search_space,
    n_calls=20,
    n_random_starts=5,
    random_state=42,
)