"""Tests for ml.config module.

Tests configuration loading, validation, and utility functions like seed setting
and config hashing.
"""

from __future__ import annotations

import hashlib
import random
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from pydantic import ValidationError

from ml.config import (
    DatasetConfig,
    FedEraseConfig,
    ModelConfig,
    TrainingConfig,
    compute_config_hash,
    load_config,
    set_seeds,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_yaml_file() -> Path:
    """Create a temporary YAML file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        temp_path = Path(f.name)
    yield temp_path
    temp_path.unlink(missing_ok=True)


@pytest.fixture
def valid_config_dict() -> dict:
    """Return a valid configuration dictionary."""
    return {
        "experiment_name": "test",
        "seed": 123,
        "dataset": {
            "name": "cifar10",
            "data_dir": "data/test",
            "val_fraction": 0.1,
            "batch_size": 32,
            "num_workers": 1,
            "augment_train": True,
        },
        "model": {
            "name": "cifar_cnn",
            "num_classes": 10,
            "conv_channels": [32, 64],
            "dropout": 0.2,
            "activation": "relu",
        },
        "training": {
            "epochs": 5,
            "optimizer": "sgd",
            "learning_rate": 0.001,
            "momentum": 0.95,
            "weight_decay": 0.0,
            "scheduler": None,
            "early_stopping_patience": None,
            "device": "cpu",
        },
        "checkpoint": {
            "checkpoint_dir": "checkpoints/test",
            "save_best_only": True,
            "keep_last_n": 2,
        },
        "logging": {
            "level": "DEBUG",
            "log_dir": "logs/test",
            "log_file": "test.log",
        },
    }


# =============================================================================
# Tests for DatasetConfig
# =============================================================================


@pytest.mark.unit
class TestDatasetConfig:
    """Tests for DatasetConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        cfg = DatasetConfig()
        assert cfg.name == "cifar10"
        assert cfg.val_fraction == 0.1
        assert cfg.batch_size == 64
        assert cfg.num_workers == 2
        assert cfg.augment_train is True

    def test_invalid_dataset_name(self) -> None:
        """Test that invalid dataset name raises error."""
        with pytest.raises(ValidationError):
            DatasetConfig(name="imagenet")

    def test_invalid_val_fraction(self) -> None:
        """Test that invalid val_fraction raises error."""
        with pytest.raises(ValidationError):
            DatasetConfig(val_fraction=1.5)

    def test_invalid_batch_size(self) -> None:
        """Test that invalid batch_size raises error."""
        with pytest.raises(ValidationError):
            DatasetConfig(batch_size=0)


# =============================================================================
# Tests for ModelConfig
# =============================================================================


@pytest.mark.unit
class TestModelConfig:
    """Tests for ModelConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        cfg = ModelConfig()
        assert cfg.name == "cifar_cnn"
        assert cfg.num_classes == 10
        assert cfg.conv_channels == [32, 64, 128]
        assert cfg.dropout == 0.3
        assert cfg.activation == "relu"

    def test_invalid_model_name(self) -> None:
        """Test that invalid model name raises error."""
        with pytest.raises(ValidationError):
            ModelConfig(name="resnet50")

    def test_invalid_activation(self) -> None:
        """Test that invalid activation raises error."""
        with pytest.raises(ValidationError):
            ModelConfig(activation="tanh")

    def test_invalid_conv_channels(self) -> None:
        """Test that invalid conv_channels raises error."""
        with pytest.raises(ValidationError):
            ModelConfig(conv_channels=[])
        with pytest.raises(ValidationError):
            ModelConfig(conv_channels=[32, -64])

    def test_valid_gelu_activation(self) -> None:
        """Test that gelu activation is valid."""
        cfg = ModelConfig(activation="gelu")
        assert cfg.activation == "gelu"


# =============================================================================
# Tests for TrainingConfig
# =============================================================================


@pytest.mark.unit
class TestTrainingConfig:
    """Tests for TrainingConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        cfg = TrainingConfig()
        assert cfg.epochs == 20
        assert cfg.optimizer == "sgd"
        assert cfg.learning_rate == 0.01
        assert cfg.scheduler == "cosine"

    def test_invalid_optimizer(self) -> None:
        """Test that invalid optimizer raises error."""
        with pytest.raises(ValidationError):
            TrainingConfig(optimizer="rmsprop")

    def test_invalid_scheduler(self) -> None:
        """Test that invalid scheduler raises error."""
        with pytest.raises(ValidationError):
            TrainingConfig(scheduler="exponential")

    def test_none_scheduler(self) -> None:
        """Test that None scheduler is valid."""
        cfg = TrainingConfig(scheduler=None)
        assert cfg.scheduler is None


# =============================================================================
# Tests for FedEraseConfig
# =============================================================================


@pytest.mark.unit
class TestFedEraseConfig:
    """Tests for FedEraseConfig."""

    def test_default_config(self) -> None:
        """Test that default config is valid."""
        cfg = FedEraseConfig()
        assert cfg.experiment_name == "phase1_baseline"
        assert cfg.seed == 42
        assert cfg.dataset.name == "cifar10"
        assert cfg.model.name == "cifar_cnn"

    def test_custom_config(self, valid_config_dict: dict) -> None:
        """Test that custom config values are accepted."""
        cfg = FedEraseConfig(**valid_config_dict)
        assert cfg.experiment_name == "test"
        assert cfg.seed == 123
        assert cfg.training.epochs == 5


# =============================================================================
# Tests for load_config
# =============================================================================


@pytest.mark.unit
class TestLoadConfig:
    """Tests for configuration loading."""

    def test_load_valid_config(self, valid_config_dict: dict, temp_yaml_file: Path) -> None:
        """Test loading a valid configuration file."""
        with open(temp_yaml_file, "w") as f:
            yaml.dump(valid_config_dict, f)

        cfg = load_config(temp_yaml_file)
        assert cfg.experiment_name == "test"
        assert cfg.seed == 123

    def test_load_missing_file(self) -> None:
        """Test that missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_config(Path("/nonexistent/config.yaml"))

    def test_load_invalid_yaml(self, temp_yaml_file: Path) -> None:
        """Test that invalid YAML raises error."""
        with open(temp_yaml_file, "w") as f:
            f.write("invalid: yaml: syntax:")

        with pytest.raises(ValueError, match="Failed to parse YAML"):
            load_config(temp_yaml_file)

    def test_load_with_invalid_nested_field_type(
        self, valid_config_dict: dict, temp_yaml_file: Path
    ) -> None:
        """Test that invalid nested field type raises ValidationError."""
        # Set a field that should be an int to a string
        valid_config_dict["training"]["epochs"] = "not_an_int"

        with open(temp_yaml_file, "w") as f:
            yaml.dump(valid_config_dict, f)

        with pytest.raises(ValidationError):
            load_config(temp_yaml_file)

    def test_load_wrong_type(self, valid_config_dict: dict, temp_yaml_file: Path) -> None:
        """Test that wrong type raises ValidationError."""
        valid_config_dict["training"]["epochs"] = "twenty"

        with open(temp_yaml_file, "w") as f:
            yaml.dump(valid_config_dict, f)

        with pytest.raises(ValidationError):
            load_config(temp_yaml_file)

    @pytest.mark.integration
    def test_load_phase1_baseline(self) -> None:
        """Test loading the actual phase1_baseline.yaml."""
        config_path = Path("experiments/configs/phase1_baseline.yaml")
        if config_path.exists():
            cfg = load_config(config_path)
            assert cfg.experiment_name == "phase1_baseline"
            assert cfg.seed == 42

    @pytest.mark.integration
    def test_load_defaults(self) -> None:
        """Test loading the defaults.yaml."""
        config_path = Path("config/defaults.yaml")
        if config_path.exists():
            cfg = load_config(config_path)
            assert cfg.dataset.name == "cifar10"


# =============================================================================
# Tests for set_seeds
# =============================================================================


@pytest.mark.unit
class TestSetSeeds:
    """Tests for seed setting."""

    def test_set_seeds_does_not_raise(self) -> None:
        """Test that set_seeds does not raise for valid seeds."""
        set_seeds(0)
        set_seeds(42)
        set_seeds(99999)

    def test_seeds_are_reproducible(self) -> None:
        """Test that same seed produces same random values."""
        set_seeds(42)
        val1_rand = random.random()
        val1_np = np.random.randn()
        val1_torch = torch.randn(1).item()

        set_seeds(42)
        val2_rand = random.random()
        val2_np = np.random.randn()
        val2_torch = torch.randn(1).item()

        assert val1_rand == val2_rand
        assert val1_np == val2_np
        assert val1_torch == val2_torch

    def test_different_seeds_produce_different_values(self) -> None:
        """Test that different seeds produce different random values."""
        set_seeds(42)
        val1 = random.random()

        set_seeds(43)
        val2 = random.random()

        assert val1 != val2


# =============================================================================
# Tests for compute_config_hash
# =============================================================================


@pytest.mark.unit
class TestComputeConfigHash:
    """Tests for config hashing."""

    def test_compute_hash_returns_64_chars(
        self, valid_config_dict: dict, temp_yaml_file: Path
    ) -> None:
        """Test that hash is a 64-character hex string."""
        with open(temp_yaml_file, "w") as f:
            yaml.dump(valid_config_dict, f)

        hash_val = compute_config_hash(temp_yaml_file)
        assert isinstance(hash_val, str)
        assert len(hash_val) == 64
        assert all(c in "0123456789abcdef" for c in hash_val)

    def test_same_file_same_hash(self, valid_config_dict: dict, temp_yaml_file: Path) -> None:
        """Test that same file produces same hash."""
        with open(temp_yaml_file, "w") as f:
            yaml.dump(valid_config_dict, f)

        hash1 = compute_config_hash(temp_yaml_file)
        hash2 = compute_config_hash(temp_yaml_file)

        assert hash1 == hash2

    def test_different_files_different_hash(
        self, valid_config_dict: dict, temp_yaml_file: Path
    ) -> None:
        """Test that different files produce different hashes."""
        with open(temp_yaml_file, "w") as f:
            yaml.dump(valid_config_dict, f)

        hash1 = compute_config_hash(temp_yaml_file)

        valid_config_dict["seed"] = 999
        with open(temp_yaml_file, "w") as f:
            yaml.dump(valid_config_dict, f)

        hash2 = compute_config_hash(temp_yaml_file)

        assert hash1 != hash2

    def test_compute_hash_missing_file(self) -> None:
        """Test that missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            compute_config_hash(Path("/nonexistent/config.yaml"))

    def test_hash_is_sha256(self, valid_config_dict: dict, temp_yaml_file: Path) -> None:
        """Test that hash matches SHA-256 of file contents."""
        with open(temp_yaml_file, "w") as f:
            yaml.dump(valid_config_dict, f)

        with open(temp_yaml_file, "rb") as f:
            file_bytes = f.read()

        expected_hash = hashlib.sha256(file_bytes).hexdigest()
        actual_hash = compute_config_hash(temp_yaml_file)

        assert actual_hash == expected_hash
