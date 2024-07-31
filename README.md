# ADS-B Anomaly Detection with Autoencoder

This project implements an autoencoder for anomaly detection in ADS-B data using TensorFlow and QKeras. It also supports converting and testing the model with HLS4ML for FPGA deployment.

## Table of Contents

1. [Project Structure](#project-structure)
2. [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Setup](#setup)
3. [Training the Model](#training-the-model)
4. [Validating the Model](#validating-the-model)
5. [Testing the Model](#testing-the-model)
6. [Testing with HLS Model](#testing-with-hls-model)

## Project Structure

```
.
├── data/
│   ├── train/
│   ├── validation/
│   ├── test_noise/
│   ├── test_landing/
│   ├── test_departing/
│   ├── test_manoeuver/
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── hls_converter.py
│   ├── model/
│   │   └── autoencoder.py
│   ├── utils/
│   │   ├── utils.py
│   │   ├── evaluation.py
│   │   ├── visualization.py
│   │   └── preprocessing.py
├── train.py
├── validate.py
├── test.py
├── convert_to_hls.py
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.8+
- Ubuntu 20.04 
- Vivado v2020.1
- Docker

### Setup

1. **Build Docker Image:**

   ```bash
   docker build -t hls4ml_adsb -f docker/Dockerfile .
   ```

2. **Run Docker Container:**

   ```bash
   docker run -p 8888:8888 -v /path/to/your/local/directory:/home/jovyan/work -it hls4ml_adsb
   ```

   Replace `/path/to/your/local/directory` with your own path.

## Training the Model

To train the autoencoder model, run:

```bash
python train.py
```

This script will:

- Load and preprocess the training data.
- Train the quantized autoencoder model.
- Save the trained model and preprocessing pipeline to the `MODEL_STANDARD_DIR`.

## Validating the Model

To validate the trained model, run:

```bash
python validate.py
```

This script will:

- Load and preprocess the validation data.
- Validate the autoencoder model.
- Calculate, save and print reconstruction error statistics.

## Testing the Model

To test the trained model, run:

```bash
python test.py
```

This script will:

- Load and preprocess the test data.
- Test the autoencoder model.
- Calculate reconstruction errors and accuracy scores.
- Generate and print a classification report.

## Testing with HLS Model

To test the model with an already converted HLS model, run:

```bash
python test.py --use_hls --hls_model_dir='hls_model/hls4ml_prj'
```

This script will:

- Load and preprocess the test data.
- Load the pre-converted HLS model from the specified directory.
- Test the HLS model.
- Calculate reconstruction errors and accuracy scores.
- Generate and print a classification report.