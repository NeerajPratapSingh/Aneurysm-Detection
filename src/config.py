"""
Centralized configuration for Aneurysm Detection project.

This module contains all configurable parameters for preprocessing,
modeling, training, and prediction.
"""

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class PreprocessingConfig:
    """Preprocessing configuration parameters."""
    
    # Image dimensions
    target_height: int = 512
    target_width: int = 512
    
    # CLAHE parameters
    clahe_clip_limit: float = 2.0
    clahe_tile_size: int = 8
    
    # Gaussian blur
    gaussian_sigma: float = 1.0
    
    # Bilateral filter
    bilateral_diameter: int = 9
    bilateral_sigma_color: float = 75.0
    bilateral_sigma_space: float = 75.0
    
    # Normalization
    normalization_method: str = "zscore"  # 'zscore' or 'minmax'
    
    # Slice selection
    slice_selection_strategy: str = "center"  # 'center', 'max_variance', 'manual'
    num_slices: int = 1


@dataclass
class CNNConfig:
    """CNN Feature Extractor configuration."""
    
    # Model architecture
    model_name: str = "resnet50"  # 'resnet50', 'vgg16', 'inception'
    pretrained: bool = True
    freeze_backbone: bool = True
    
    # Input
    input_channels: int = 3
    input_size: Tuple[int, int] = (224, 224)
    
    # Output features
    feature_dim: int = 2048
    
    # Training
    batch_size: int = 32
    learning_rate: float = 0.001
    num_epochs: int = 10
    device: str = "cuda"  # 'cuda' or 'cpu'


@dataclass
class BoostingConfig:
    """Boosting Classifier configuration."""
    
    # Model type
    model_type: str = "gradient_boosting"  # 'gradient_boosting' or 'xgboost'
    
    # Hyperparameters
    n_estimators: int = 100
    learning_rate: float = 0.1
    max_depth: int = 5
    min_samples_split: int = 5
    min_samples_leaf: int = 2
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    random_state: int = 42
    
    # Validation
    cv_folds: int = 5
    validation_split: float = 0.2


@dataclass
class BaselineConfig:
    """Baseline Model configuration."""
    
    # Model type
    model_type: str = "logistic_regression"  # 'logistic_regression', 'knn', 'random_forest'
    
    # Hyperparameters (varies by model)
    max_iter: int = 1000  # For Logistic Regression
    n_neighbors: int = 5  # For KNN
    n_estimators: int = 100  # For Random Forest
    
    random_state: int = 42


@dataclass
class PipelineConfig:
    """Pipeline configuration."""
    
    # Pipeline type
    pipeline_type: str = "hybrid"  # 'hybrid' or 'original'
    
    # Feature extraction
    feature_extractor: str = "cnn"  # 'cnn' or 'hand_crafted'
    cnn_model: str = "resnet50"
    
    # Classification
    classifier: str = "gradient_boosting"  # 'gradient_boosting', 'xgboost', 'random_forest'
    
    # Training
    random_state: int = 42
    test_size: float = 0.2
    stratified: bool = True
    
    # Cross-validation
    use_cv: bool = True
    cv_folds: int = 5


@dataclass
class DataConfig:
    """Data configuration."""
    
    # Paths
    data_dir: str = "data/"
    metadata_path: str = "data/metadata.csv"
    sample_dir: str = "data/sample/"
    output_dir: str = "outputs/"
    
    # Data parameters
    test_size: float = 0.2
    val_size: float = 0.1
    random_state: int = 42


@dataclass
class AppConfig:
    """Web Application configuration."""
    
    # Server
    host: str = "localhost"
    port: int = 8501
    debug: bool = False
    
    # UI
    max_upload_size_mb: int = 100
    allowed_formats: List[str] = None
    
    def __post_init__(self):
        if self.allowed_formats is None:
            self.allowed_formats = [".dcm", ".DCM", ".jpg", ".png"]


class Config:
    """Master configuration class combining all sub-configs."""
    
    def __init__(self):
        self.preprocessing = PreprocessingConfig()
        self.cnn = CNNConfig()
        self.boosting = BoostingConfig()
        self.baseline = BaselineConfig()
        self.pipeline = PipelineConfig()
        self.data = DataConfig()
        self.app = AppConfig()
    
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            "preprocessing": self.preprocessing.__dict__,
            "cnn": self.cnn.__dict__,
            "boosting": self.boosting.__dict__,
            "baseline": self.baseline.__dict__,
            "pipeline": self.pipeline.__dict__,
            "data": self.data.__dict__,
            "app": self.app.__dict__,
        }


# Global config instance
config = Config()
