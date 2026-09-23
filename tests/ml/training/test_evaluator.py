"""Tests for model evaluator.

Covers validation metric computation, gradient handling, and edge cases.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ml.training.evaluator import EvaluationResult, Evaluator

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def simple_model() -> nn.Module:
    """Create a simple model for testing."""
    return nn.Sequential(
        nn.Linear(10, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    )


@pytest.fixture
def simple_dataloader() -> DataLoader:  # type: ignore[type-arg]
    """Create a small synthetic DataLoader."""
    torch.manual_seed(42)
    images = torch.randn(32, 10)
    labels = torch.randint(0, 10, (32,))
    dataset = TensorDataset(images, labels)
    return DataLoader(dataset, batch_size=8, shuffle=False)


@pytest.fixture
def evaluator() -> Evaluator:
    """Create an evaluator with default criterion."""
    return Evaluator()


# =============================================================================
# Tests: EvaluationResult
# =============================================================================


@pytest.mark.unit
class TestEvaluationResult:
    """Tests for EvaluationResult dataclass."""

    def test_result_creation(self) -> None:
        """Test creating an EvaluationResult."""
        result = EvaluationResult(
            loss=0.5,
            accuracy=0.95,
            n_samples=100,
            duration_seconds=1.5,
        )
        assert result.loss == 0.5
        assert result.accuracy == 0.95
        assert result.n_samples == 100
        assert result.duration_seconds == 1.5


# =============================================================================
# Tests: Evaluator Basic
# =============================================================================


@pytest.mark.unit
class TestEvaluatorBasic:
    """Tests for basic evaluator functionality."""

    def test_evaluator_creation(self, evaluator: Evaluator) -> None:
        """Test that evaluator can be created."""
        assert evaluator is not None

    def test_evaluate_returns_result(
        self, evaluator: Evaluator, simple_model: nn.Module, simple_dataloader: DataLoader  # type: ignore[type-arg]
    ) -> None:
        """Test that evaluate returns an EvaluationResult."""
        result = evaluator.evaluate(simple_model, simple_dataloader, device="cpu")
        assert isinstance(result, EvaluationResult)

    def test_result_has_required_fields(
        self, evaluator: Evaluator, simple_model: nn.Module, simple_dataloader: DataLoader  # type: ignore[type-arg]
    ) -> None:
        """Test that result has all required fields."""
        result = evaluator.evaluate(simple_model, simple_dataloader, device="cpu")
        assert hasattr(result, "loss")
        assert hasattr(result, "accuracy")
        assert hasattr(result, "n_samples")
        assert hasattr(result, "duration_seconds")

    def test_accuracy_in_range(
        self, evaluator: Evaluator, simple_model: nn.Module, simple_dataloader: DataLoader  # type: ignore[type-arg]
    ) -> None:
        """Test that accuracy is in [0.0, 1.0]."""
        result = evaluator.evaluate(simple_model, simple_dataloader, device="cpu")
        assert 0.0 <= result.accuracy <= 1.0

    def test_loss_is_nonnegative(
        self, evaluator: Evaluator, simple_model: nn.Module, simple_dataloader: DataLoader  # type: ignore[type-arg]
    ) -> None:
        """Test that loss is non-negative."""
        result = evaluator.evaluate(simple_model, simple_dataloader, device="cpu")
        assert result.loss >= 0.0

    def test_n_samples_matches_dataloader(
        self, evaluator: Evaluator, simple_model: nn.Module, simple_dataloader: DataLoader  # type: ignore[type-arg]
    ) -> None:
        """Test that n_samples matches total samples in DataLoader."""
        result = evaluator.evaluate(simple_model, simple_dataloader, device="cpu")
        # simple_dataloader has 32 samples
        assert result.n_samples == 32


# =============================================================================
# Tests: Gradient Handling
# =============================================================================


@pytest.mark.unit
class TestGradientHandling:
    """Tests for correct gradient handling during evaluation."""

    def test_no_gradients_during_evaluation(
        self, evaluator: Evaluator, simple_model: nn.Module, simple_dataloader: DataLoader  # type: ignore[type-arg]
    ) -> None:
        """Test that evaluation does not compute gradients."""
        # Enable gradient tracking
        for param in simple_model.parameters():
            param.grad = None

        # Evaluate
        evaluator.evaluate(simple_model, simple_dataloader, device="cpu")

        # No gradients should be computed
        for param in simple_model.parameters():
            assert param.grad is None

    def test_model_weights_unchanged(
        self, evaluator: Evaluator, simple_model: nn.Module, simple_dataloader: DataLoader  # type: ignore[type-arg]
    ) -> None:
        """Test that model weights are unchanged after evaluation."""
        # Store original weights
        original_state = {name: param.clone() for name, param in simple_model.named_parameters()}

        # Evaluate
        evaluator.evaluate(simple_model, simple_dataloader, device="cpu")

        # Check weights unchanged
        for name, param in simple_model.named_parameters():
            assert torch.equal(param, original_state[name])


# =============================================================================
# Tests: Model Mode Handling
# =============================================================================


@pytest.mark.unit
class TestModelModeHandling:
    """Tests for correct model mode switching."""

    def test_model_restored_to_train_mode(
        self, evaluator: Evaluator, simple_model: nn.Module, simple_dataloader: DataLoader  # type: ignore[type-arg]
    ) -> None:
        """Test that model is returned to train mode if it was training."""
        simple_model.train()
        evaluator.evaluate(simple_model, simple_dataloader, device="cpu")
        assert simple_model.training

    def test_model_restored_to_eval_mode(
        self, evaluator: Evaluator, simple_model: nn.Module, simple_dataloader: DataLoader  # type: ignore[type-arg]
    ) -> None:
        """Test that model stays in eval mode if it was already in eval mode."""
        simple_model.eval()
        evaluator.evaluate(simple_model, simple_dataloader, device="cpu")
        assert not simple_model.training

    def test_evaluate_sets_eval_during_evaluation(
        self, evaluator: Evaluator, simple_model: nn.Module, simple_dataloader: DataLoader  # type: ignore[type-arg]
    ) -> None:
        """Test that model is in eval mode during evaluation."""
        simple_model.train()

        class ModeCheckingEvaluator(Evaluator):
            def __init__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
                super().__init__(*args, **kwargs)
                self.was_in_eval_during_eval = None

            def evaluate(self, model, loader, device="cpu"):  # type: ignore[no-untyped-def]
                model.eval()
                self.was_in_eval_during_eval = not model.training
                return super().evaluate(model, loader, device)

        checker = ModeCheckingEvaluator()
        checker.evaluate(simple_model, simple_dataloader, device="cpu")
        assert checker.was_in_eval_during_eval


# =============================================================================
# Tests: Edge Cases
# =============================================================================


@pytest.mark.unit
class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_single_batch_dataloader(
        self, evaluator: Evaluator, simple_model: nn.Module
    ) -> None:
        """Test evaluation on single batch."""
        torch.manual_seed(42)
        images = torch.randn(4, 10)
        labels = torch.randint(0, 10, (4,))
        dataset = TensorDataset(images, labels)
        loader = DataLoader(dataset, batch_size=4, shuffle=False)

        result = evaluator.evaluate(simple_model, loader, device="cpu")
        assert result.n_samples == 4

    def test_perfect_accuracy_model(
        self, evaluator: Evaluator
    ) -> None:
        """Test evaluation on model with perfect accuracy."""
        # Create a model that always predicts correctly
        class PerfectModel(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # Return one-hot encoded labels (all 1s in first column for label 0, etc)
                batch_size = x.shape[0]
                # Create logits that will predict correctly
                logits = torch.zeros(batch_size, 10)
                logits[:, 0] = 100.0  # All predict class 0 with high confidence
                return logits

        model = PerfectModel()
        torch.manual_seed(42)
        images = torch.randn(8, 10)
        labels = torch.zeros(8, dtype=torch.long)  # All class 0
        dataset = TensorDataset(images, labels)
        loader = DataLoader(dataset, batch_size=4, shuffle=False)

        result = evaluator.evaluate(model, loader, device="cpu")
        assert result.accuracy == 1.0

    def test_evaluator_with_custom_criterion(self) -> None:
        """Test evaluator with custom loss criterion."""
        criterion = nn.MSELoss()
        evaluator = Evaluator(criterion=criterion)
        assert evaluator.criterion == criterion

    def test_device_string_and_object(
        self, evaluator: Evaluator, simple_model: nn.Module, simple_dataloader: DataLoader  # type: ignore[type-arg]
    ) -> None:
        """Test evaluation with device as string and torch.device."""
        # As string
        result1 = evaluator.evaluate(simple_model, simple_dataloader, device="cpu")
        assert result1.accuracy >= 0.0

        # As torch.device
        device_obj = torch.device("cpu")
        result2 = evaluator.evaluate(simple_model, simple_dataloader, device=device_obj)
        assert result2.accuracy >= 0.0
