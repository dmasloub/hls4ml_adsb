# optimization.py

from skopt import gp_minimize
from skopt.space import Integer, Real, Categorical
from skopt.utils import use_named_args
import warnings

from src.data_preparation import prepare_data
from train import train_autoencoder
from validate import validate_autoencoder
from test import test_autoencoder

# Optional: Visualize the optimization process
from skopt.plots import plot_convergence, plot_objective
import pickle

import matplotlib.pyplot as plt

# Suppress any warnings for cleaner output
warnings.filterwarnings("ignore")

# Define the search space
search_space = [
    Categorical([4, 6, 8], name="bits"),
    Categorical([0 ,2], name="integer"),
    Real(0.4, 0.95, name='pruning_percent'),
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
    pruning_percent = params['pruning_percent']
    begin_step = params['begin_step']
    frequency = params['frequency']

    # Train the autoencoder with given parameters
    train_autoencoder(
        preprocessed_data=preprocessed_data_param,
        pruning_percent=pruning_percent,
        begin_step=begin_step,
        frequency=frequency,
        bits=bits,
        integer=integer
    )

    # Validate the autoencoder to get reconstruction error statistics
    mu, std = validate_autoencoder(preprocessed_data=preprocessed_data_param)

    # Test the autoencoder to get accuracy and resource utilization
    accuracy, utilization = test_autoencoder(preprocessed_data=preprocessed_data_param, build_hls_model=True)

    # Get the LUT utilization
    total_lut = utilization.get('Total_LUT', 0) if utilization else float('inf')

    # Combine accuracy and utilization into a single score
    score = accuracy - lambda_reg * total_lut

    return -score  # Negative because most optimizers minimize the objective function


# Now define the objective function wrapper to include preprocessed_data
@use_named_args(search_space)
def objective(**params):
    return objective_function(params, preprocessed_data)


# Run the optimization
res = gp_minimize(
    func=objective,
    dimensions=search_space,
    n_calls=20,  # Increased to 20 for better optimization
    n_random_starts=5,  # Increased to 5 for better exploration
    random_state=42,
)


# After optimization, output the optimal values
def print_optimal_results(res, search_space):
    # Extract parameter names from the search space
    parameter_names = [dim.name for dim in search_space]

    # Create a dictionary of optimal parameters
    optimal_parameters = dict(zip(parameter_names, res.x))

    # Print the optimal parameters
    print("\nOptimal Parameters Found:")
    for name, value in optimal_parameters.items():
        print(f"  {name}: {value}")

    # Print the best score (remember to negate it back)
    best_score = -res.fun
    print(f"\nBest Score: {best_score}")

    # Optionally, print additional information
    print("\nAdditional Optimization Details:")
    print(f"  Number of calls: {res.func_vals.shape[0]}")
    print(f"  Best function value: {res.fun}")
    print(f"  Location of best function value: {res.x}")


# Call the function to print results
print_optimal_results(res, search_space)

with open('optimization_results.pkl', 'wb') as f:
    pickle.dump(res, f)

print("\nOptimization results have been saved to 'optimization_results.pkl'.")

# Plot convergence and save to file
fig1 = plot_convergence(res)
fig1.savefig('convergence_plot.png')

# Plot objective and save to file
fig2 = plot_objective(res)
fig2.savefig('objective_plot.png')

print("Convergence and objective plots have been saved as 'convergence_plot.png' and 'objective_plot.png'.")
