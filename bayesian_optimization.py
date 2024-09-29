# bayesian_optimization.py

import logging
import sys
import os
import numpy as np  # Import NumPy for array operations
from skopt import gp_minimize
from skopt.space import Integer, Real, Categorical
from skopt.utils import use_named_args
from skopt.callbacks import CheckpointSaver
import warnings

from src.data_preparation import prepare_data
from train import train_autoencoder
from validate import validate_autoencoder
from test import test_autoencoder

# Optional: Visualize the optimization process
from skopt.plots import plot_convergence, plot_objective
import pickle

import matplotlib.pyplot as plt

# Define the total number of optimization calls
TOTAL_CALLS = 50

# Suppress any warnings for cleaner output
warnings.filterwarnings("ignore")

# -------------------------------
# Configure the logger
# -------------------------------
logger = logging.getLogger('OptimizationLogger')
logger.setLevel(logging.INFO)  # Set the desired log level

# Create formatter - includes timestamp, log level, and message
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# Create console handler and set level to info
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)

# Create file handler with rotating logs to prevent unlimited growth
from logging.handlers import RotatingFileHandler

fh = RotatingFileHandler('optimization.log', maxBytes=5*1024*1024, backupCount=5)  # 5 MB per file, keep 5 backups
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)

# Add handlers to the logger
if not logger.handlers:
    logger.addHandler(ch)
    logger.addHandler(fh)

# -------------------------------
# Define the search space
# -------------------------------
search_space = [
    Categorical([4, 6, 8], name="bits"),
    Categorical([0, 2], name="integer"),
    Real(0.1, 5, name="alpha"),
    Real(0.4, 0.7, name='pruning_percent'),
    Integer(1000, 5000, name='begin_step'),
    Integer(200, 500, name='frequency'),
]

# Path to save/load preprocessed data
preprocessed_data_path = 'preprocessed_data.pkl'

# Prepare the data once (load if exists, else preprocess and save)
preprocessed_data = prepare_data(save_path=preprocessed_data_path, load_if_exists=True)

def objective_function(params, preprocessed_data_param, lambda_reg=0.5):
    # Extract parameters
    integer = params['integer']
    bits = params['bits']
    alpha = params['alpha']
    pruning_percent = params['pruning_percent']
    begin_step = params['begin_step']
    frequency = params['frequency']

    # Log the current set of parameters being evaluated
    logger.info(f"Evaluating Parameters: bits={bits}, integer={integer}, alpha={alpha}, "
                f"pruning_percent={pruning_percent}, begin_step={begin_step}, frequency={frequency}")

    try:
        # Train the autoencoder with given parameters
        train_autoencoder(
            preprocessed_data=preprocessed_data_param,
            pruning_percent=pruning_percent,
            begin_step=begin_step,
            frequency=frequency,
            bits=bits,
            integer=integer,
            alpha=alpha,
        )

        # Validate the autoencoder to get reconstruction error statistics
        mu, std = validate_autoencoder(preprocessed_data=preprocessed_data_param)

        # Test the autoencoder to get accuracy and resource utilization
        accuracy, utilization = test_autoencoder(preprocessed_data=preprocessed_data_param, build_hls_model=True)

        # Get the LUT utilization
        total_lut = utilization.get('Total_LUT', 0) if utilization else float('inf')
        util_lut = utilization.get('Utilization (%)_LUT', 0) if utilization else 100

        # Get the FF utilization
        total_ff = utilization.get('Total_FF', 0) if utilization else float('inf')
        util_ff = utilization.get('Utilization (%)_FF', 0) if utilization else 100

        # Get the DSP utilization
        total_dsp = utilization.get('Total_DSP48E', 0) if utilization else float('inf')
        util_dsp = utilization.get('Utilization (%)_DSP48E', 0) if utilization else 100

        if util_lut >= 100 or util_dsp >= 100 or util_ff >= 100:
            # If any utilization is maxed out, assign a very bad score
            score = -1
            logger.info(f"Utilization exceeded limits: LUT={util_lut}%, FF={util_ff}%, DSP48E={util_dsp}%. Assigned score={score}")
        else:
            # Combine accuracy and utilization into a single score
            # Normalize LUT, DSP, and FF utilizations based on their maximum values
            normalized_lut = total_lut / 53200
            normalized_dsp = total_dsp / 220
            normalized_ff = total_ff / 106400
            average_util = (normalized_lut + normalized_dsp + normalized_ff) / 3
            score = accuracy - lambda_reg * average_util

            # Log the results of the evaluation
            logger.info(f"Result: accuracy={accuracy}, util_LUT={util_lut}, util_FF={util_ff}, "
                        f"util_DSP48E={util_dsp}, score={score}")

        return -score  # Negative because most optimizers minimize the objective function

    except Exception as e:
        # Log any exceptions that occur during the evaluation
        logger.error(f"Error during evaluation with parameters {params}: {e}", exc_info=True)
        return float('inf')  # Assign a poor score in case of failure

# Now define the objective function wrapper to include preprocessed_data
@use_named_args(search_space)
def objective(**params):
    return objective_function(params, preprocessed_data)

# Define a callback function to monitor optimization progress
def print_callback(res):
    # res is an OptimizeResult object
    iteration = len(res.func_vals)
    current_best = -np.min(res.func_vals)  # Since we are minimizing -score

    # Use np.argmin to get the index of the minimum value
    best_index = np.argmin(res.func_vals)
    best_params = res.x_iters[best_index]

    logger.info(f"Iteration {iteration}: Current Best Score = {current_best}")
    logger.info(f"Best Parameters So Far: bits={best_params[0]}, integer={best_params[1]}, "
                f"alpha={best_params[2]}, pruning_percent={best_params[3]}, "
                f"begin_step={best_params[4]}, frequency={best_params[5]}\n")

# Optionally, use CheckpointSaver to save optimization progress
checkpoint_file = "optimization_checkpoint.pkl"
checkpoint_saver = CheckpointSaver(checkpoint_file, compress=9)

# -------------------------------
# Resuming Optimization Logic
# -------------------------------
def load_checkpoint(checkpoint_path):
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'rb') as f:
            res = pickle.load(f)
        logger.info(f"Loaded checkpoint from '{checkpoint_path}'.")
        return res
    else:
        logger.info(f"No checkpoint found at '{checkpoint_path}'. Starting fresh optimization.")
        return None

def get_initial_data(res):
    if res is not None:
        x0 = res.x_iters
        y0 = res.func_vals
        return x0, y0
    else:
        return None, None

# Load existing checkpoint if available
existing_res = load_checkpoint(checkpoint_file)

# Extract initial data if checkpoint exists
x0, y0 = get_initial_data(existing_res)

# Calculate the number of calls already done
n_calls_done = len(x0) if x0 else 0

# Determine the number of calls remaining
n_calls_remaining = TOTAL_CALLS - n_calls_done

if n_calls_remaining <= 0:
    logger.info(f"Already completed {n_calls_done} trials, which meets or exceeds TOTAL_CALLS={TOTAL_CALLS}.")
    logger.info("Skipping optimization.")
    # Optionally, load the best parameters from the checkpoint
    if existing_res:
        print_optimal_results(existing_res, search_space)
    sys.exit(0)

logger.info(f"Starting optimization: {n_calls_remaining} trials remaining out of {TOTAL_CALLS} total.")

# Run the optimization with the callback and checkpoint saver
res = gp_minimize(
    func=objective,
    dimensions=search_space,
    n_calls=n_calls_remaining,
    x0=x0,
    y0=y0,
    n_random_starts=max(5 - n_calls_done, 0),  # Ensure at least 5 random starts
    random_state=42,
    callback=[print_callback, checkpoint_saver],  # Add the callback here
)

# After optimization, output the optimal values
def print_optimal_results(res, search_space):
    # Extract parameter names from the search space
    parameter_names = [dim.name for dim in search_space]

    # Create a dictionary of optimal parameters
    optimal_parameters = dict(zip(parameter_names, res.x))

    # Log the optimal parameters
    logger.info("\nOptimal Parameters Found:")
    for name, value in optimal_parameters.items():
        logger.info(f"  {name}: {value}")

    # Log the best score (remember to negate it back)
    best_score = -res.fun
    logger.info(f"\nBest Score: {best_score}")

    # Optionally, log additional information
    logger.info("\nAdditional Optimization Details:")
    logger.info(f"  Number of calls: {res.func_vals.shape[0]}")
    logger.info(f"  Best function value: {res.fun}")
    logger.info(f"  Location of best function value: {res.x}")

# Call the function to print results
print_optimal_results(res, search_space)

# Save the optimization results
with open('optimization_results.pkl', 'wb') as f:
    pickle.dump(res, f)

logger.info("\nOptimization results have been saved to 'optimization_results.pkl'.")

# Plot convergence and save to file
fig1 = plot_convergence(res)
fig1.savefig('convergence_plot.png')

# Plot objective and save to file
fig2 = plot_objective(res)
fig2.savefig('objective_plot.png')

logger.info("Convergence and objective plots have been saved as 'convergence_plot.png' and 'objective_plot.png'.")
