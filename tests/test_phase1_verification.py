"""Phase 1 experimental verification tests.

Verifies reproducibility, checkpoint integrity, dataset determinism,
and system correctness without requiring long training runs.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ml.config import load_config, set_seeds
from ml.datasets.cifar10 import CIFAR10DataModule
from ml.models.cnn import CifarCNN, ModelFactory
from ml.training.checkpoint import CheckpointManager, CheckpointMetadata
from ml.training.evaluator import Evaluator
from ml.training.trainer import Trainer

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_results_dir() -> Path:
    """Create temporary directory for results."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def config_seed_42():
    """Load baseline config with seed 42."""
    config = load_config(Path("experiments/configs/phase1_baseline.yaml"))
    assert config.seed == 42
    return config


@pytest.fixture
def config_seed_123(config_seed_42):
    """Create config with seed 123."""
    # Modify config to use seed 123
    config = config_seed_42
    config.seed = 123
    return config


@pytest.fixture
def simple_model():
    """Create a simple model for testing."""
    return nn.Sequential(
        nn.Linear(32, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    )


@pytest.fixture
def simple_loader():
    """Create a simple DataLoader for testing."""
    torch.manual_seed(42)
    x = torch.randn(64, 32)
    y = torch.randint(0, 10, (64,))
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=8, shuffle=False)


# =============================================================================
# Test: Reproducibility - Same Seed Produces Same Results
# =============================================================================


@pytest.mark.unit
class TestReproducibilitySameSeed:
    """Verify that identical seeds produce identical training metrics."""

    def test_dataset_split_reproducibility(self) -> None:
        """Test that same seed produces consistent dataset setup."""
        config = load_config(Path("experiments/configs/phase1_baseline.yaml"))

        # Set seeds and get loaders twice
        set_seeds(config.seed)
        dataset1 = CIFAR10DataModule(config.dataset)
        dataset1.prepare_data()
        dataset1.setup()

        set_seeds(config.seed)
        dataset2 = CIFAR10DataModule(config.dataset)
        dataset2.prepare_data()
        dataset2.setup()

        # Both should have same size loaders
        assert len(dataset1.train_loader) == len(dataset2.train_loader)  # type: ignore[arg-type]
        assert len(dataset1.val_loader) == len(dataset2.val_loader)  # type: ignore[arg-type]

    def test_model_initialization_reproducibility(self) -> None:
        """Test that same seed produces same initial model parameters."""
        config = load_config(Path("experiments/configs/phase1_baseline.yaml"))

        # Create model 1 with seed 42
        set_seeds(42)
        factory1 = ModelFactory()
        model1 = factory1.create(config.model)
        params1 = {k: v.clone() for k, v in model1.state_dict().items()}

        # Create model 2 with seed 42
        set_seeds(42)
        factory2 = ModelFactory()
        model2 = factory2.create(config.model)
        params2 = {k: v.clone() for k, v in model2.state_dict().items()}

        # Parameters should be identical
        for key in params1.keys():
            assert torch.allclose(
                params1[key], params2[key]
            ), f"Model parameters differ for key {key}"


# =============================================================================
# Test: Different Seed Produces Different Results
# =============================================================================


@pytest.mark.unit
class TestDifferentSeedDifferentResults:
    """Verify that different seeds produce different splits/initializations."""

    def test_different_seed_different_split(self) -> None:
        """Test that different seeds produce different train/val splits."""
        config1 = load_config(Path("experiments/configs/phase1_baseline.yaml"))
        config2 = load_config(Path("experiments/configs/phase1_baseline.yaml"))

        # Seed 42
        set_seeds(42)
        dataset1 = CIFAR10DataModule(config1.dataset)
        dataset1.prepare_data()
        dataset1.setup()
        batch1 = next(iter(dataset1.train_loader))

        # Seed 123
        set_seeds(123)
        dataset2 = CIFAR10DataModule(config2.dataset)
        dataset2.prepare_data()
        dataset2.setup()
        batch2 = next(iter(dataset2.train_loader))

        # Batches should differ (different seed should give different split)
        # Note: may have some overlap, but not identical
        assert not torch.allclose(
            batch1[0], batch2[0], atol=1e-2
        ), "Different seeds produced identical batches (unlikely)"

    def test_different_seed_different_model_params(self) -> None:
        """Test that different seeds produce different model initializations."""
        config = load_config(Path("experiments/configs/phase1_baseline.yaml"))

        # Model with seed 42
        set_seeds(42)
        model1 = ModelFactory().create(config.model)
        params1 = model1.state_dict()

        # Model with seed 123
        set_seeds(123)
        model2 = ModelFactory().create(config.model)
        params2 = model2.state_dict()

        # At least some parameters should differ
        diff_count = 0
        for key in params1.keys():
            if not torch.allclose(params1[key], params2[key]):
                diff_count += 1

        assert diff_count > 0, "Different seeds produced identical model parameters"


# =============================================================================
# Test: Checkpoint Integrity and Verification
# =============================================================================


@pytest.mark.unit
class TestCheckpointIntegrity:
    """Verify checkpoint save/load and tamper detection."""

    def test_checkpoint_save_load_cycle(
        self, temp_results_dir: Path, simple_model: nn.Module
    ) -> None:
        """Test complete checkpoint save/load cycle."""
        manager = CheckpointManager(temp_results_dir)

        # Create metadata
        metadata = CheckpointMetadata(
            model_name="test_model",
            epoch=5,
            val_accuracy=0.95,
            val_loss=0.15,
            total_epochs_trained=5,
            config_hash="test_hash_123",
            model_architecture=str(simple_model),
            pytorch_version=torch.__version__,
            checkpoint_version=1,
        )

        # Save checkpoint
        checkpoint_path = manager.save(simple_model, metadata, name="test")
        assert checkpoint_path.exists(), "Checkpoint file not created"

        # Verify metadata file exists
        meta_path = checkpoint_path.parent / checkpoint_path.name.replace(".pt", ".meta.json")
        assert meta_path.exists(), "Metadata file not created"

        # Verify checksum file exists
        checksum_path = checkpoint_path.parent / checkpoint_path.name.replace(".pt", ".sha256")
        assert checksum_path.exists(), "Checksum file not created"

        # Load checkpoint
        model2 = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )
        loaded_metadata = manager.load(model2, checkpoint_path)

        # Verify metadata matches
        assert loaded_metadata.model_name == "test_model"
        assert loaded_metadata.epoch == 5
        assert loaded_metadata.val_accuracy == 0.95

    def test_checkpoint_tamper_detection(
        self, temp_results_dir: Path, simple_model: nn.Module
    ) -> None:
        """Test that tampering with checkpoint is detected."""
        manager = CheckpointManager(temp_results_dir)

        metadata = CheckpointMetadata(
            model_name="test_model",
            epoch=1,
            val_accuracy=0.50,
            val_loss=0.50,
            total_epochs_trained=1,
            config_hash="hash1",
            model_architecture=str(simple_model),
            pytorch_version=torch.__version__,
            checkpoint_version=1,
        )

        checkpoint_path = manager.save(simple_model, metadata, name="test")

        # Tamper with checkpoint by appending data
        with checkpoint_path.open("ab") as f:
            f.write(b"TAMPERED")

        # Attempt to load should fail with checksum mismatch
        model2 = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

        with pytest.raises(ValueError, match="Checksum mismatch"):
            manager.load(model2, checkpoint_path)

    def test_checkpoint_missing_file_error(
        self, temp_results_dir: Path, simple_model: nn.Module
    ) -> None:
        """Test error when checkpoint file is missing."""
        manager = CheckpointManager(temp_results_dir)
        missing_path = temp_results_dir / "nonexistent.pt"

        with pytest.raises(FileNotFoundError, match="Checkpoint file not found"):
            manager.load(simple_model, missing_path)


# =============================================================================
# Test: Configuration Validation
# =============================================================================


@pytest.mark.unit
class TestConfigurationValidation:
    """Verify that configuration is properly loaded and validated."""

    def test_baseline_config_loads(self) -> None:
        """Test that baseline config loads without errors."""
        config = load_config(Path("experiments/configs/phase1_baseline.yaml"))
        assert config.seed == 42
        assert config.experiment_name == "phase1_baseline"
        assert config.dataset.name == "cifar10"
        assert config.model.name == "cifar_cnn"

    def test_quick_test_config_loads(self) -> None:
        """Test that quick test config loads."""
        config = load_config(Path("experiments/configs/phase1_quick_test.yaml"))
        assert config.seed == 42
        assert config.training.epochs == 2
        assert config.model.conv_channels == [32, 64]

    def test_model_factory_creates_model(self) -> None:
        """Test that ModelFactory creates model from config."""
        config = load_config(Path("experiments/configs/phase1_baseline.yaml"))
        factory = ModelFactory()
        model = factory.create(config.model)

        assert isinstance(model, CifarCNN)
        assert model.config.num_classes == 10

    def test_model_factory_works(self) -> None:
        """Test that ModelFactory successfully creates models."""
        config = load_config(Path("experiments/configs/phase1_baseline.yaml"))
        factory = ModelFactory()
        model = factory.create(config.model)

        assert isinstance(model, CifarCNN)
        assert model.config.num_classes == 10


# =============================================================================
# Test: Trainer and Evaluator Basics
# =============================================================================


@pytest.mark.unit
class TestTrainerEvaluatorBasics:
    """Verify Trainer and Evaluator functionality."""

    def test_evaluator_computes_metrics(
        self, simple_model: nn.Module, simple_loader: DataLoader
    ) -> None:
        """Test that Evaluator computes accuracy and loss."""
        evaluator = Evaluator()
        result = evaluator.evaluate(simple_model, simple_loader, "cpu")

        assert 0.0 <= result.accuracy <= 1.0, "Accuracy out of range"
        assert result.loss >= 0.0, "Loss is negative"
        assert result.n_samples == 64, "Sample count mismatch"

    def test_trainer_one_epoch(
        self,
        simple_model: nn.Module,
        simple_loader: DataLoader,
    ) -> None:
        """Test that Trainer runs one epoch without errors."""
        from ml.config import TrainingConfig

        config = TrainingConfig(
            epochs=1,
            optimizer="sgd",
            learning_rate=0.01,
            momentum=0.9,
            weight_decay=0.0,
            scheduler=None,
            early_stopping_patience=None,
            device="cpu",
        )

        trainer = Trainer(config)
        result = trainer.train(
            simple_model,
            simple_loader,
            simple_loader,
            experiment_name="test",
        )

        assert result.total_epochs == 1
        assert result.best_epoch == 1
        assert result.final_train_loss >= 0.0


# =============================================================================
# Test: Results Storage
# =============================================================================


@pytest.mark.unit
class TestResultsStorage:
    """Verify experiment results storage structure."""

    def test_results_directory_creation(self, temp_results_dir: Path) -> None:
        """Test that results directory structure is created."""
        results_dir = temp_results_dir / "phase1" / "experiment_001"
        results_dir.mkdir(parents=True, exist_ok=True)

        assert results_dir.exists()
        assert results_dir.is_dir()

    def test_results_json_storage(self, temp_results_dir: Path) -> None:
        """Test that results can be stored as JSON."""
        results = {
            "metadata": {"seed": 42, "experiment": "test"},
            "results": {"accuracy": 0.95, "loss": 0.15},
            "timing": {"duration": 100.5},
        }

        results_file = temp_results_dir / "results.json"
        with results_file.open("w") as f:
            json.dump(results, f, indent=2)

        # Verify it can be read back
        with results_file.open("r") as f:
            loaded_results = json.load(f)

        assert loaded_results == results
        assert loaded_results["metadata"]["seed"] == 42


# =============================================================================
# Integration Test
# =============================================================================


@pytest.mark.integration
class TestPhase1Integration:
    """End-to-end integration test for Phase 1 components."""

    def test_full_pipeline_synthetic_data(
        self, temp_results_dir: Path
    ) -> None:
        """Test complete pipeline with synthetic data."""
        # Load config
        config = load_config(Path("experiments/configs/phase1_baseline.yaml"))
        assert config.seed == 42

        # Set seed
        set_seeds(config.seed)

        # Create model
        model = ModelFactory().create(config.model)
        assert isinstance(model, CifarCNN)

        # Create synthetic loaders (avoid network download)
        torch.manual_seed(config.seed)
        x_train = torch.randn(128, 3, 32, 32)
        y_train = torch.randint(0, 10, (128,))
        train_dataset = TensorDataset(x_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        x_val = torch.randn(64, 3, 32, 32)
        y_val = torch.randint(0, 10, (64,))
        val_dataset = TensorDataset(x_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        # Use CPU device explicitly
        config.training.device = "cpu"

        # Train model
        trainer = Trainer(config.training)
        training_result = trainer.train(
            model,
            train_loader,
            val_loader,
            experiment_name=config.experiment_name,
        )

        assert training_result.total_epochs >= 1
        assert training_result.best_val_accuracy >= 0.0
        assert training_result.best_val_accuracy <= 1.0

        # Evaluate on val set
        evaluator = Evaluator()
        test_result = evaluator.evaluate(model, val_loader, torch.device("cpu"))

        assert 0.0 <= test_result.accuracy <= 1.0
        assert test_result.loss >= 0.0

        # Save checkpoint
        manager = CheckpointManager(temp_results_dir)
        metadata = CheckpointMetadata(
            model_name=config.model.name,
            epoch=training_result.best_epoch,
            val_accuracy=training_result.best_val_accuracy,
            val_loss=test_result.loss,
            total_epochs_trained=training_result.total_epochs,
            config_hash="",
            model_architecture=str(model),
            pytorch_version=torch.__version__,
            checkpoint_version=1,
        )

        checkpoint_path = manager.save(model, metadata, name="best")
        assert checkpoint_path.exists()
