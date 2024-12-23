# src/config/config.py

from dataclasses import dataclass, field
from typing import List

@dataclass
class PathsConfig:
    data_train_dir: str = "data/train/"
    data_validation_dir: str = "data/validation/"
    data_test_noise_dir: str = "data/test/noise/"
    data_test_landing_dir: str = "data/test/landing/"
    data_test_departing_dir: str = "data/test/departing/"
    data_test_manoeuver_dir: str = "data/test/manoeuver/"
    model_dir: str = "qkeras_model"
    hls_output_dir: str = "hls_model/"
    logs_dir: str = "logs/"
    checkpoints_dir: str = "checkpoints/"
    preprocessed_data_path: str = "preprocessed_data.pkl"

@dataclass
class DataConfig:
    features: List[str] = field(default_factory=lambda: ["longitude", "latitude", "altitude", "groundspeed", "x", "y"])
    window_size: int = 60
    diff_data: bool = True
    diff_features: List[str] = field(default_factory=lambda: ["longitude", "latitude", "x", "y"])
    k_lag: int = 1
    k_order: int = 1
    std_threshold_train: float = 5.0
    std_threshold_validation: float = 8.0

@dataclass
class ModelConfig:
    # input_dim: int = 6                
    encoding_dim: int = 10
    bits: int = 4                     
    integer_bits: int = 2
    alpha: float = 10.0
    learning_rate: float = 0.001
    batch_size: int = 128
    epochs: int = 50
    standard_q_threshold: float = 0.3892
    pruning_percent: float = 0.9467
    begin_step: int = 5000
    frequency: int = 268


@dataclass
class OptimizationConfig:
    total_calls: int = 100
    random_state: int = 42
    n_initial_points: int = 10
    lambda_reg: float = 0.5
    penalty_score: float = 1e6
    search_space: dict = field(default_factory=lambda: {
        "bits": [4, 6, 8],
        "integer": [0, 2, 4],
        "alpha": (1.0, 10.0),
        "pruning_percent": (0.0, 0.95),
        "standard_q_threshold": (0.01, 0.5),
        "begin_step": (0, 5000),
        "frequency": (200, 500)
    })

@dataclass
class HLSConfig:
    backend: str = "VivadoAccelerator"
    board: str = "pynq-z2"
    granularity: str = "models"

@dataclass
class Config:
    paths: PathsConfig = PathsConfig()
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    optimization: OptimizationConfig = OptimizationConfig()
    hls: HLSConfig = HLSConfig()
