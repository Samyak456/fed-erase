"""Configuration management for FedErase.

Provides Pydantic models for all runtime settings, configuration loading with
validation, and utility functions for reproducibility (seed setting, config hashing).

All configuration is loaded from YAML files — no hard-coded values in code.
"""

from __future__ import annotations

import hashlib
import logging
import random
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import yaml
from pydantic import BaseModel, Field, field_validator

# Configure logging for this module
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration Models
# ---------------------------------------------------------------------------


class DatasetConfig(BaseModel):
    """Configuration for dataset loading and preprocessing.

    Attributes:
        name: Dataset name (currently "cifar10" only).
        data_dir: Directory where dataset is downloaded/cached.
        val_fraction: Fraction of training set reserved for validation (0 to 1).
        batch_size: DataLoader batch size.
        num_workers: Number of DataLoader worker processes.
        augment_train: Whether to apply augmentation (crop, flip) to training data.

    """

    name: str = Field(default="cifar10", description="Dataset name")
    data_dir: Path = Field(default=Path("data/cifar10"), description="Data directory")
    val_fraction: float = Field(
        default=0.1, ge=0.0, le=1.0, description="Validation fraction of training set"
    )
    batch_size: int = Field(default=64, gt=0, description="Batch size")
    num_workers: int = Field(default=2, ge=0, description="DataLoader worker processes")
    augment_train: bool = Field(default=True, description="Enable train augmentation")

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Ensure dataset name is supported."""
        if v != "cifar10":
            raise ValueError(f'Only "cifar10" is supported, got "{v}"')
        return v


class ModelConfig(BaseModel):
    """Configuration for model architecture.

    Attributes:
        name: Model name (currently "cifar_cnn" only).
        num_classes: Number of output classes (10 for CIFAR-10).
        conv_channels: List of filter counts per convolutional block.
        dropout: Dropout rate before final FC layer.
        activation: Activation function name ("relu" or "gelu").

    """

    name: str = Field(default="cifar_cnn", description="Model name")
    num_classes: int = Field(default=10, gt=0, description="Number of output classes")
    conv_channels: list[int] = Field(
        default=[32, 64, 128],
        description="Filter counts per conv block",
    )
    dropout: float = Field(
        default=0.3, ge=0.0, le=1.0, description="Dropout rate"
    )
    activation: Literal["relu", "gelu"] = Field(
        default="relu", description="Activation function"
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Ensure model name is supported."""
        if v != "cifar_cnn":
            raise ValueError(f'Only "cifar_cnn" is supported, got "{v}"')
        return v

    @field_validator("conv_channels")
    @classmethod
    def validate_conv_channels(cls, v: list[int]) -> list[int]:
        """Ensure at least one conv block."""
        if not v or any(c <= 0 for c in v):
            raise ValueError("conv_channels must be non-empty list of positive integers")
        return v


class TrainingConfig(BaseModel):
    """Configuration for training loop.

    Attributes:
        epochs: Number of training epochs.
        optimizer: Optimizer name ("sgd" or "adam").
        learning_rate: Learning rate.
        momentum: Momentum (for SGD).
        weight_decay: L2 regularization weight.
        scheduler: Learning rate scheduler name (or None for no scheduler).
        early_stopping_patience: Number of non-improving epochs before stopping (or None).
        device: Device selection ("auto", "cpu", or "cuda").

    """

    epochs: int = Field(default=20, ge=0, description="Number of epochs")
    optimizer: Literal["sgd", "adam"] = Field(default="sgd", description="Optimizer")
    learning_rate: float = Field(default=0.01, gt=0, description="Learning rate")
    momentum: float = Field(default=0.9, ge=0.0, le=1.0, description="Momentum for SGD")
    weight_decay: float = Field(
        default=1e-4, ge=0.0, description="L2 regularization weight"
    )
    scheduler: Literal["cosine", "step"] | None = Field(
        default="cosine", description="LR scheduler name"
    )
    early_stopping_patience: int | None = Field(
        default=7, ge=1, description="Early stopping patience"
    )
    device: Literal["auto", "cpu", "cuda"] = Field(
        default="auto", description="Device selection"
    )


class CheckpointConfig(BaseModel):
    """Configuration for model checkpointing.

    Attributes:
        checkpoint_dir: Directory where checkpoints are saved.
        save_best_only: Whether to save only the best model.
        keep_last_n: Number of recent checkpoints to keep (for cleanup).

    """

    checkpoint_dir: Path = Field(
        default=Path("checkpoints/phase1"), description="Checkpoint directory"
    )
    save_best_only: bool = Field(
        default=True, description="Save only the best model by validation accuracy"
    )
    keep_last_n: int = Field(default=3, gt=0, description="Keep N recent checkpoints")


class LoggingConfig(BaseModel):
    """Configuration for logging.

    Attributes:
        level: Logging level ("DEBUG", "INFO", "WARNING", "ERROR").
        log_dir: Directory for log files.
        log_file: Name of the log file.

    """

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO", description="Logging level"
    )
    log_dir: Path = Field(default=Path("logs"), description="Log directory")
    log_file: str = Field(default="phase1.log", description="Log file name")


class FedEraseConfig(BaseModel):
    """Top-level configuration for FedErase Phase 1.

    Attributes:
        experiment_name: Name of the experiment (used for logging, checkpoints, etc.).
        seed: Random seed for reproducibility.
        dataset: Dataset configuration.
        model: Model configuration.
        training: Training configuration.
        checkpoint: Checkpoint configuration.
        logging: Logging configuration.

    """

    experiment_name: str = Field(
        default="phase1_baseline", description="Experiment name"
    )
    seed: int = Field(default=42, description="Random seed")
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    class Config:
        """Pydantic config."""

        validate_assignment = True
        arbitrary_types_allowed = True


# ---------------------------------------------------------------------------
# Configuration Loading and Utilities
# ---------------------------------------------------------------------------


def load_config(path: Path) -> FedEraseConfig:
    """Load and validate configuration from a YAML file.

    Sets random seeds immediately upon loading.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        FedEraseConfig instance.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ValidationError: If the configuration is invalid (from Pydantic).
        yaml.YAMLError: If the YAML file cannot be parsed.

    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    logger.debug(f"Loading configuration from {path}")

    try:
        with path.open() as f:
            config_dict = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Failed to parse YAML configuration: {e}") from e

    # Convert nested dicts to their appropriate Pydantic models
    config = FedEraseConfig(**config_dict)

    # Set seeds immediately
    set_seeds(config.seed)
    logger.info(f"Configuration loaded from {path}, seed set to {config.seed}")

    return config


def set_seeds(seed: int) -> None:
    """Set all random seeds for reproducibility.

    Sets:
    - Python's random module
    - NumPy's random state
    - PyTorch's random state (CPU and CUDA if available)

    Args:
        seed: Random seed value.

    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    logger.debug(f"All random seeds set to {seed}")


def compute_config_hash(path: Path) -> str:
    """Compute SHA-256 hash of a configuration file.

    Used to track which config was used for a checkpoint.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        Hex-encoded SHA-256 hash (64 characters).

    Raises:
        FileNotFoundError: If the file does not exist.

    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("rb") as f:
        file_bytes = f.read()

    hash_obj = hashlib.sha256(file_bytes)
    return hash_obj.hexdigest()


# ---------------------------------------------------------------------------
# Configuration Utilities for Paths
# ---------------------------------------------------------------------------


def resolve_path(base_path: Path, relative_path: Path) -> Path:
    """Resolve a relative path against a base path.

    If `relative_path` is absolute, return it as-is. Otherwise, resolve it
    relative to `base_path`.

    Args:
        base_path: Base path (typically the project root).
        relative_path: Path to resolve (may be relative or absolute).

    Returns:
        Resolved absolute path.

    """
    if relative_path.is_absolute():
        return relative_path
    return (base_path / relative_path).resolve()
