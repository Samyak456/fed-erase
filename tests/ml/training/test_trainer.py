"""Tests for training loop.

Covers trainer functionality, epoch tracking, optimizer handling, and edge cases.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ml.config import TrainingConfig
from ml.training.trainer import (
    EpochRecord,
    Trainer,
    TrainingResult,
    _build_optimizer,
    _build_scheduler,
    _resolve_device,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def simple_model() -> nn.Module:
    """Create a simple model for training."""
    return nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 10),
    )


@pytest.fixture
def tiny_train_loader() -> DataLoader:  # type: ignore[type-arg]
    """Create a tiny training DataLoader."""
    torch.manual_seed(42)
    images = torch.randn(16, 10)
    labels = torch.randint(0, 10, (16,))
    dataset = TensorDataset(images, labels)
    return DataLoader(dataset, batch_size=8, shuffle=True)


@pytest.fixture
def tiny_val_loader() -> DataLoader:  # type: ignore[type-arg]
    """Create a tiny validation DataLoader."""
    torch.manual_seed(42)
    images = torch.randn(8, 10)
    labels = torch.randint(0, 10, (8,))
    dataset = TensorDataset(images, labels)
    return DataLoader(dataset, batch_size=4, shuffle=False)


@pytest.fixture
def training_config() -> TrainingConfig:
    """Create a training config for testing."""
    return TrainingConfig(
        epochs=2,
        optimizer="sgd",
        learning_rate=0.01,
        momentum=0.9,
        weight_decay=0.0001,
        scheduler="cosine",
        early_stopping_patience=None,
        device="cpu",
    )


# =============================================================================
# Tests: Device Resolution
# =============================================================================


@pytest.mark.unit
class TestDeviceResolution:
    """Tests for device resolution."""

    def test_resolve_device_auto_returns_device(self) -> None:
        """Test that resolve_device('auto') returns a torch.device."""
        device = _resolve_device("auto")
        assert isinstance(device, torch.device)

    def test_resolve_device_cpu_returns_cpu(self) -> None:
        """Test that resolve_device('cpu') returns CPU device."""
        device = _resolve_device("cpu")
        assert device.type == "cpu"

    def test_resolve_device_cuda_raises_if_unavailable(self) -> None:
        """Test that resolve_device('cuda') raises if CUDA not available."""
        if not torch.cuda.is_available():
            with pytest.raises(ValueError, match="CUDA requested but not available"):
                _resolve_device("cuda")
        else:
            device = _resolve_device("cuda")
            assert device.type == "cuda"

    def test_resolve_device_invalid_raises(self) -> None:
        """Test that invalid device raises ValueError."""
        with pytest.raises(ValueError, match="Invalid device config"):
            _resolve_device("invalid")


# =============================================================================
# Tests: Optimizer Building
# =============================================================================


@pytest.mark.unit
class TestOptimizerBuilding:
    """Tests for optimizer building."""

    def test_build_sgd_optimizer(self, simple_model: nn.Module) -> None:
        """Test building SGD optimizer."""
        config = TrainingConfig(
            epochs=1,
            optimizer="sgd",
            learning_rate=0.01,
            momentum=0.9,
            weight_decay=0.0001,
            scheduler=None,
            early_stopping_patience=None,
            device="cpu",
        )
        optimizer = _build_optimizer(simple_model, config)
        assert isinstance(optimizer, torch.optim.SGD)

    def test_build_adam_optimizer(self, simple_model: nn.Module) -> None:
        """Test building Adam optimizer."""
        config = TrainingConfig(
            epochs=1,
            optimizer="adam",
            learning_rate=0.001,
            momentum=0.0,
            weight_decay=0.0001,
            scheduler=None,
            early_stopping_patience=None,
            device="cpu",
        )
        optimizer = _build_optimizer(simple_model, config)
        assert isinstance(optimizer, torch.optim.Adam)

    def test_build_invalid_optimizer_raises(self, simple_model: nn.Module) -> None:
        """Test that Pydantic validates optimizer name."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="Input should be 'sgd' or 'adam'"):
            TrainingConfig(
                epochs=1,
                optimizer="rmsprop",  # type: ignore
                learning_rate=0.01,
                momentum=0.9,
                weight_decay=0.0001,
                scheduler=None,
                early_stopping_patience=None,
                device="cpu",
            )


# =============================================================================
# Tests: Scheduler Building
# =============================================================================


@pytest.mark.unit
class TestSchedulerBuilding:
    """Tests for scheduler building."""

    def test_build_cosine_scheduler(self, simple_model: nn.Module) -> None:
        """Test building CosineAnnealingLR scheduler."""
        optimizer = torch.optim.SGD(simple_model.parameters(), lr=0.01)
        config = TrainingConfig(
            epochs=10,
            optimizer="sgd",
            learning_rate=0.01,
            momentum=0.9,
            weight_decay=0.0001,
            scheduler="cosine",
            early_stopping_patience=None,
            device="cpu",
        )
        scheduler = _build_scheduler(optimizer, config)
        assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)

    def test_build_step_scheduler(self, simple_model: nn.Module) -> None:
        """Test building StepLR scheduler."""
        optimizer = torch.optim.SGD(simple_model.parameters(), lr=0.01)
        config = TrainingConfig(
            epochs=10,
            optimizer="sgd",
            learning_rate=0.01,
            momentum=0.9,
            weight_decay=0.0001,
            scheduler="step",
            early_stopping_patience=None,
            device="cpu",
        )
        scheduler = _build_scheduler(optimizer, config)
        assert isinstance(scheduler, torch.optim.lr_scheduler.StepLR)

    def test_build_none_scheduler(self, simple_model: nn.Module) -> None:
        """Test that None scheduler returns None."""
        optimizer = torch.optim.SGD(simple_model.parameters(), lr=0.01)
        config = TrainingConfig(
            epochs=10,
            optimizer="sgd",
            learning_rate=0.01,
            momentum=0.9,
            weight_decay=0.0001,
            scheduler=None,
            early_stopping_patience=None,
            device="cpu",
        )
        scheduler = _build_scheduler(optimizer, config)
        assert scheduler is None


# =============================================================================
# Tests: Training Basic
# =============================================================================


@pytest.mark.unit
class TestTrainingBasic:
    """Tests for basic training functionality."""

    def test_trainer_creation(self, training_config: TrainingConfig) -> None:
        """Test that trainer can be created."""
        trainer = Trainer(training_config)
        assert trainer is not None

    def test_training_completes(
        self,
        simple_model: nn.Module,
        tiny_train_loader: DataLoader,  # type: ignore[type-arg]
        tiny_val_loader: DataLoader,  # type: ignore[type-arg]
        training_config: TrainingConfig,
    ) -> None:
        """Test that training completes without error."""
        trainer = Trainer(training_config)
        result = trainer.train(simple_model, tiny_train_loader, tiny_val_loader)
        assert isinstance(result, TrainingResult)

    def test_result_has_required_fields(
        self,
        simple_model: nn.Module,
        tiny_train_loader: DataLoader,  # type: ignore[type-arg]
        tiny_val_loader: DataLoader,  # type: ignore[type-arg]
        training_config: TrainingConfig,
    ) -> None:
        """Test that result has all required fields."""
        trainer = Trainer(training_config)
        result = trainer.train(simple_model, tiny_train_loader, tiny_val_loader)
        assert hasattr(result, "experiment_name")
        assert hasattr(result, "total_epochs")
        assert hasattr(result, "best_val_accuracy")
        assert hasattr(result, "history")
        assert hasattr(result, "total_duration_seconds")
        assert hasattr(result, "device")


# =============================================================================
# Tests: Training History
# =============================================================================


@pytest.mark.unit
class TestTrainingHistory:
    """Tests for training history tracking."""

    def test_history_has_one_record_per_epoch(
        self,
        simple_model: nn.Module,
        tiny_train_loader: DataLoader,  # type: ignore[type-arg]
        tiny_val_loader: DataLoader,  # type: ignore[type-arg]
        training_config: TrainingConfig,
    ) -> None:
        """Test that history has one EpochRecord per epoch."""
        trainer = Trainer(training_config)
        result = trainer.train(simple_model, tiny_train_loader, tiny_val_loader)
        assert len(result.history) == training_config.epochs

    def test_epoch_record_has_required_fields(
        self,
        simple_model: nn.Module,
        tiny_train_loader: DataLoader,  # type: ignore[type-arg]
        tiny_val_loader: DataLoader,  # type: ignore[type-arg]
        training_config: TrainingConfig,
    ) -> None:
        """Test that EpochRecord has all required fields."""
        trainer = Trainer(training_config)
        result = trainer.train(simple_model, tiny_train_loader, tiny_val_loader)
        record = result.history[0]
        assert isinstance(record, EpochRecord)
        assert hasattr(record, "epoch")
        assert hasattr(record, "train_loss")
        assert hasattr(record, "train_accuracy")
        assert hasattr(record, "val_loss")
        assert hasattr(record, "val_accuracy")
        assert hasattr(record, "duration_seconds")

    def test_accuracy_in_valid_range(
        self,
        simple_model: nn.Module,
        tiny_train_loader: DataLoader,  # type: ignore[type-arg]
        tiny_val_loader: DataLoader,  # type: ignore[type-arg]
        training_config: TrainingConfig,
    ) -> None:
        """Test that accuracies are in [0.0, 1.0]."""
        trainer = Trainer(training_config)
        result = trainer.train(simple_model, tiny_train_loader, tiny_val_loader)
        for record in result.history:
            assert 0.0 <= record.train_accuracy <= 1.0
            assert 0.0 <= record.val_accuracy <= 1.0

    def test_loss_is_nonnegative(
        self,
        simple_model: nn.Module,
        tiny_train_loader: DataLoader,  # type: ignore[type-arg]
        tiny_val_loader: DataLoader,  # type: ignore[type-arg]
        training_config: TrainingConfig,
    ) -> None:
        """Test that losses are non-negative."""
        trainer = Trainer(training_config)
        result = trainer.train(simple_model, tiny_train_loader, tiny_val_loader)
        for record in result.history:
            assert record.train_loss >= 0.0
            assert record.val_loss >= 0.0


# =============================================================================
# Tests: Parameter Updates
# =============================================================================


@pytest.mark.unit
class TestParameterUpdates:
    """Tests for model parameter changes during training."""

    def test_model_parameters_change(
        self,
        simple_model: nn.Module,
        tiny_train_loader: DataLoader,  # type: ignore[type-arg]
        tiny_val_loader: DataLoader,  # type: ignore[type-arg]
        training_config: TrainingConfig,
    ) -> None:
        """Test that model parameters change after training."""
        # Store original parameters
        original_params = {name: param.clone() for name, param in simple_model.named_parameters()}

        # Train
        trainer = Trainer(training_config)
        trainer.train(simple_model, tiny_train_loader, tiny_val_loader)

        # Check parameters changed
        params_changed = False
        for name, param in simple_model.named_parameters():
            if not torch.equal(param, original_params[name]):
                params_changed = True
                break
        assert params_changed


# =============================================================================
# Tests: Early Stopping
# =============================================================================


@pytest.mark.unit
class TestEarlyStopping:
    """Tests for early stopping functionality."""

    def test_early_stopping_stops_training(
        self,
        simple_model: nn.Module,
        tiny_train_loader: DataLoader,  # type: ignore[type-arg]
        tiny_val_loader: DataLoader,  # type: ignore[type-arg]
    ) -> None:
        """Test that early stopping stops training before max epochs."""
        config = TrainingConfig(
            epochs=100,  # Large number
            optimizer="sgd",
            learning_rate=0.01,
            momentum=0.9,
            weight_decay=0.0001,
            scheduler=None,
            early_stopping_patience=1,  # Stop after 1 epoch without improvement
            device="cpu",
        )
        trainer = Trainer(config)
        result = trainer.train(simple_model, tiny_train_loader, tiny_val_loader)
        # Should stop before 100 epochs
        assert result.total_epochs < config.epochs


# =============================================================================
# Tests: Edge Cases
# =============================================================================


@pytest.mark.unit
class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_zero_epochs(
        self,
        simple_model: nn.Module,
        tiny_train_loader: DataLoader,  # type: ignore[type-arg]
        tiny_val_loader: DataLoader,  # type: ignore[type-arg]
    ) -> None:
        """Test training with zero epochs."""
        config = TrainingConfig(
            epochs=0,
            optimizer="sgd",
            learning_rate=0.01,
            momentum=0.9,
            weight_decay=0.0001,
            scheduler=None,
            early_stopping_patience=None,
            device="cpu",
        )
        trainer = Trainer(config)
        result = trainer.train(simple_model, tiny_train_loader, tiny_val_loader)
        assert result.total_epochs == 0

    def test_single_epoch(
        self,
        simple_model: nn.Module,
        tiny_train_loader: DataLoader,  # type: ignore[type-arg]
        tiny_val_loader: DataLoader,  # type: ignore[type-arg]
    ) -> None:
        """Test training with single epoch."""
        config = TrainingConfig(
            epochs=1,
            optimizer="sgd",
            learning_rate=0.01,
            momentum=0.9,
            weight_decay=0.0001,
            scheduler=None,
            early_stopping_patience=None,
            device="cpu",
        )
        trainer = Trainer(config)
        result = trainer.train(simple_model, tiny_train_loader, tiny_val_loader)
        assert result.total_epochs == 1
        assert len(result.history) == 1

    def test_training_without_scheduler(
        self,
        simple_model: nn.Module,
        tiny_train_loader: DataLoader,  # type: ignore[type-arg]
        tiny_val_loader: DataLoader,  # type: ignore[type-arg]
    ) -> None:
        """Test that training works without scheduler."""
        config = TrainingConfig(
            epochs=2,
            optimizer="sgd",
            learning_rate=0.01,
            momentum=0.9,
            weight_decay=0.0001,
            scheduler=None,
            early_stopping_patience=None,
            device="cpu",
        )
        trainer = Trainer(config)
        result = trainer.train(simple_model, tiny_train_loader, tiny_val_loader)
        assert result.total_epochs == 2
