# Quantized Autoencoder for Anomaly Detection in ADS-B Data

This repository contains the code for a scientific paper focused on designing a quantized autoencoder for anomaly detection in Automatic Dependent Surveillance-Broadcast (ADS-B) data. The project leverages techniques such as data preprocessing, model quantization, pruning, and hardware acceleration using High-Level Synthesis (HLS) for FPGA deployment.

## Table of Contents

- [Background](#background)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
  - [Data Preparation](#data-preparation)
  - [Training the Model](#training-the-model)
  - [Validation](#validation)
  - [Testing](#testing)
  - [Optimization](#optimization)
- [HLS Conversion and FPGA Deployment](#hls-conversion-and-fpga-deployment)
- [Evaluation and Visualization](#evaluation-and-visualization)
- [Contributing](#contributing)
- [License](#license)

## Background

Anomaly detection in ADS-B data is crucial for aviation safety and security. ADS-B is a surveillance technology in which an aircraft determines its position via satellite navigation and periodically broadcasts it, enabling it to be tracked. However, ADS-B lacks authentication mechanisms, making it susceptible to spoofing and other malicious activities.

This project aims to develop a quantized autoencoder capable of detecting anomalies in ADS-B data while being optimized for deployment on FPGA hardware. By quantizing the model and employing pruning techniques, we reduce the model's size and computational requirements, making it suitable for real-time applications on resource-constrained devices.

## Project Structure

```
├── src
│   ├── config
│   │   └── config.py
│   ├── converters
│   │   └── hls_converter.py
│   ├── data
│   │   ├── data_loader.py
│   │   └── data_preparation.py
│   ├── evaluation
│   │   └── evaluator.py
│   ├── models
│   │   └── autoencoder.py
│   ├── optimizers
│   │   └── bayesian_optimizer.py
│   ├── scripts
│   │   ├── optimize.py
│   │   ├── test.py
│   │   ├── train.py
│   │   ├── validate.py
│   │   └── train_val_test.py
│   ├── utils
│   │   ├── common_utils.py
│   │   ├── evaluation.py
│   │   ├── hls_utils.py
│   │   ├── logger.py
│   │   └── visualization.py
│   └── __init__.py
└── README.md
```

- **config**: Configuration files and classes.
- **converters**: Contains the HLS converter for converting Keras models to HLS.
- **data**: Data loading and preprocessing modules.
- **evaluation**: Evaluation utilities and metrics computation.
- **models**: Definition of the quantized autoencoder model.
- **optimizers**: Bayesian optimization for hyperparameter tuning.
- **scripts**: Executable scripts for training, testing, validation, and optimization.
- **utils**: Utility functions for logging, common operations, and visualization.

## Getting Started

### Prerequisites

- **Python 3.7** or higher
- Virtual environment tools (e.g., `venv` or `conda`)
- Required Python packages (see `requirements.txt`)
- **FPGA synthesis tools** (optional, for HLS conversion):
  - [Xilinx Vivado HLS](https://www.xilinx.com/products/design-tools/vivado.html)
  - [hls4ml](https://fastmachinelearning.org/hls4ml/)

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your_username/adsb-anomaly-detection.git
   cd adsb-anomaly-detection
   ```

2. **Create and activate a virtual environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install required packages:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up FPGA tools (optional):**

   - Install Xilinx Vivado HLS if you plan to perform HLS conversion and FPGA deployment.
   - Install `hls4ml`:

     ```bash
     pip install hls4ml
     ```

## Usage

### Data Preparation

The first step is to prepare the dataset for training, validation, and testing.

1. **Place your raw ADS-B data in the appropriate directories:**

   ```
   data/
   ├── train/
   ├── validation/
   └── test/
       ├── noise/
       ├── landing/
       ├── departing/
       └── manoeuver/
   ```

2. **Run the data preparation script:**

   The data preparation includes steps such as outlier removal, differencing, windowing, and feature extraction using `tsfresh`.

   ```bash
   python src/scripts/data_preparation.py
   ```

   *Note: Ensure that the configuration in `src/config/config.py` matches your data paths and preprocessing settings.*

### Training the Model

Train the quantized autoencoder model using the prepared data.

```bash
python src/scripts/train.py
```

- The model employs quantization using `QKeras` and pruning via the TensorFlow Model Optimization Toolkit.
- Training logs and checkpoints will be saved in the `logs/` and `checkpoints/` directories, respectively.

### Validation

Validate the trained model on the validation dataset to compute reconstruction errors and determine the threshold for anomaly detection.

```bash
python src/scripts/validate.py
```

- The script computes the mean and standard deviation of reconstruction errors, which are used in the anomaly scoring function.

### Testing

Evaluate the model's performance on test datasets.

```bash
python src/scripts/test.py
```

- The script runs the model on different test sets (e.g., `noise`, `landing`, `departing`, `manoeuver`) and computes evaluation metrics such as accuracy, precision, recall, and average detection delay.

### Optimization

Perform Bayesian optimization to fine-tune the model's hyperparameters for better performance and resource utilization.

```bash
python src/scripts/optimize.py
```

- The optimizer adjusts hyperparameters like quantization bits, pruning percentage, and others to minimize the objective function, which considers both model performance and hardware resource usage.
- Optimization results and progress are saved for analysis.

## HLS Conversion and FPGA Deployment

Convert the trained model to HLS code suitable for FPGA deployment using `hls4ml`.

```bash
python src/scripts/hls_conversion.py
```

- The `HLSConverter` class handles the conversion process.
- FPGA resource utilization metrics are extracted from the synthesis report.

*Note: Ensure that you have Xilinx Vivado HLS installed and properly configured in your environment.*

## Evaluation and Visualization

Generate plots and reports to visualize training progress, reconstruction errors, optimization results, and resource utilization.

- **Training History Plot:**

  After training, a plot of the loss and metrics over epochs is generated.

- **Reconstruction Error Distribution:**

  Visualize the distribution of reconstruction errors to understand the model's ability to reconstruct normal data.

- **Optimization Progress:**

  Plot the progression of Bayesian optimization to see how hyperparameters affect performance and resource usage.

- **Resource Utilization:**

  Visualize the FPGA resource utilization (LUTs, FFs, DSPs) to ensure the model fits within hardware constraints.

*Plots are saved in the `logs/` directory.*
