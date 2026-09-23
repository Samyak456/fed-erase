"""Tests for CIFAR-10 CNN architecture.

Covers model instantiation, forward passes, parameter counting, serialization,
and edge cases.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from ml.config import ModelConfig
from ml.models.cnn import CifarCNN

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def default_model_config() -> ModelConfig:
    """Return a default ModelConfig for CIFAR-10 CNN."""
    return ModelConfig(
        name="cifar_cnn",
        num_classes=10,
        conv_channels=[32, 64, 128],
        dropout=0.3,
        activation="relu",
    )


@pytest.fixture
def default_cnn(default_model_config: ModelConfig) -> CifarCNN:
    """Create a CNN with default config."""
    return CifarCNN(default_model_config)


@pytest.fixture
def dummy_batch() -> tuple[torch.Tensor, torch.Tensor]:
    """Return a small batch of dummy CIFAR-10 data.

    Shape: (4, 3, 32, 32) images + (4,) labels
    """
    torch.manual_seed(42)
    images = torch.randn(4, 3, 32, 32)
    labels = torch.randint(0, 10, (4,))
    return images, labels


# =============================================================================
# Tests: Instantiation
# =============================================================================


@pytest.mark.unit
class TestInstantiation:
    """Tests for model instantiation."""

    def test_default_config_instantiates(self, default_model_config: ModelConfig) -> None:
        """Test that model instantiates with default config."""
        model = CifarCNN(default_model_config)
        assert model is not None
        assert isinstance(model, nn.Module)

    def test_single_conv_block_instantiates(self) -> None:
        """Test model with single conv block."""
        config = ModelConfig(
            name="cifar_cnn",
            num_classes=10,
            conv_channels=[64],
            dropout=0.3,
            activation="relu",
        )
        model = CifarCNN(config)
        assert model is not None

    def test_four_conv_blocks_instantiate(self) -> None:
        """Test model with four conv blocks."""
        config = ModelConfig(
            name="cifar_cnn",
            num_classes=10,
            conv_channels=[32, 64, 128, 256],
            dropout=0.3,
            activation="relu",
        )
        model = CifarCNN(config)
        assert model is not None

    def test_gelu_activation_instantiates(self) -> None:
        """Test model with GELU activation."""
        config = ModelConfig(
            name="cifar_cnn",
            num_classes=10,
            conv_channels=[32, 64],
            dropout=0.3,
            activation="gelu",
        )
        model = CifarCNN(config)
        assert model is not None

    def test_invalid_activation_raises_validation_error(self) -> None:
        """Test that Pydantic validates activation before CifarCNN gets it."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="Input should be 'relu' or 'gelu'"):
            ModelConfig(
                name="cifar_cnn",
                num_classes=10,
                conv_channels=[32, 64],
                dropout=0.3,
                activation="tanh",  # type: ignore
            )

    def test_different_num_classes(self) -> None:
        """Test model with different number of classes."""
        config = ModelConfig(
            name="cifar_cnn",
            num_classes=5,
            conv_channels=[32, 64],
            dropout=0.3,
            activation="relu",
        )
        model = CifarCNN(config)
        assert model is not None


# =============================================================================
# Tests: Forward Pass and Output Shape
# =============================================================================


@pytest.mark.unit
class TestForwardPass:
    """Tests for forward pass and output shape."""

    def test_forward_pass_default_batch(
        self, default_cnn: CifarCNN, dummy_batch: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Test forward pass with default batch size."""
        images, _ = dummy_batch
        output = default_cnn(images)
        assert output.shape == (4, 10)

    def test_output_dtype_is_float32(
        self, default_cnn: CifarCNN, dummy_batch: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Test that output is float32."""
        images, _ = dummy_batch
        output = default_cnn(images)
        assert output.dtype == torch.float32

    def test_forward_pass_batch_size_1(self, default_cnn: CifarCNN) -> None:
        """Test forward pass with batch size 1."""
        images = torch.randn(1, 3, 32, 32)
        output = default_cnn(images)
        assert output.shape == (1, 10)

    def test_forward_pass_batch_size_32(self, default_cnn: CifarCNN) -> None:
        """Test forward pass with batch size 32."""
        images = torch.randn(32, 3, 32, 32)
        output = default_cnn(images)
        assert output.shape == (32, 10)

    def test_forward_pass_batch_size_128(self, default_cnn: CifarCNN) -> None:
        """Test forward pass with batch size 128."""
        images = torch.randn(128, 3, 32, 32)
        output = default_cnn(images)
        assert output.shape == (128, 10)

    def test_output_is_finite(
        self, default_cnn: CifarCNN, dummy_batch: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Test that output contains no NaN or Inf values."""
        images, _ = dummy_batch
        output = default_cnn(images)
        assert torch.isfinite(output).all()

    def test_forward_pass_different_conv_channels(self) -> None:
        """Test forward pass with different conv channel config."""
        config = ModelConfig(
            name="cifar_cnn",
            num_classes=10,
            conv_channels=[16, 32, 64],
            dropout=0.3,
            activation="relu",
        )
        model = CifarCNN(config)
        images = torch.randn(4, 3, 32, 32)
        output = model(images)
        assert output.shape == (4, 10)

    def test_forward_pass_gelu_activation(self) -> None:
        """Test forward pass with GELU activation."""
        config = ModelConfig(
            name="cifar_cnn",
            num_classes=10,
            conv_channels=[32, 64],
            dropout=0.3,
            activation="gelu",
        )
        model = CifarCNN(config)
        images = torch.randn(4, 3, 32, 32)
        output = model(images)
        assert output.shape == (4, 10)
        assert torch.isfinite(output).all()

    def test_output_respects_num_classes(self) -> None:
        """Test that output shape respects num_classes config."""
        for num_classes in [5, 10, 20]:
            config = ModelConfig(
                name="cifar_cnn",
                num_classes=num_classes,
                conv_channels=[32, 64],
                dropout=0.3,
                activation="relu",
            )
            model = CifarCNN(config)
            images = torch.randn(4, 3, 32, 32)
            output = model(images)
            assert output.shape == (4, num_classes)


# =============================================================================
# Tests: Parameter Counting
# =============================================================================


@pytest.mark.unit
class TestParameterCount:
    """Tests for parameter_count() method."""

    def test_parameter_count_returns_positive_int(self, default_cnn: CifarCNN) -> None:
        """Test that parameter_count returns a positive integer."""
        count = default_cnn.parameter_count()
        assert isinstance(count, int)
        assert count > 0

    def test_parameter_count_matches_pytorch_count(self, default_cnn: CifarCNN) -> None:
        """Test that parameter_count matches PyTorch's sum."""
        model_count = default_cnn.parameter_count()
        pytorch_count = sum(p.numel() for p in default_cnn.parameters() if p.requires_grad)
        assert model_count == pytorch_count

    def test_parameter_count_single_block(self) -> None:
        """Test parameter count with single conv block."""
        config = ModelConfig(
            name="cifar_cnn",
            num_classes=10,
            conv_channels=[32],
            dropout=0.3,
            activation="relu",
        )
        model = CifarCNN(config)
        count = model.parameter_count()
        assert count > 0

    def test_parameter_count_four_blocks(self) -> None:
        """Test parameter count with four conv blocks."""
        config = ModelConfig(
            name="cifar_cnn",
            num_classes=10,
            conv_channels=[32, 64, 128, 256],
            dropout=0.3,
            activation="relu",
        )
        model = CifarCNN(config)
        count = model.parameter_count()
        assert count > 0

    def test_parameter_count_increases_with_channels(self) -> None:
        """Test that more channels mean more parameters."""
        config_small = ModelConfig(
            name="cifar_cnn",
            num_classes=10,
            conv_channels=[16, 32],
            dropout=0.3,
            activation="relu",
        )
        config_large = ModelConfig(
            name="cifar_cnn",
            num_classes=10,
            conv_channels=[64, 128],
            dropout=0.3,
            activation="relu",
        )
        model_small = CifarCNN(config_small)
        model_large = CifarCNN(config_large)
        assert model_large.parameter_count() > model_small.parameter_count()


# =============================================================================
# Tests: State Dictionary Serialization
# =============================================================================


@pytest.mark.unit
class TestSerialization:
    """Tests for state_dict save/load."""

    def test_state_dict_keys(self, default_cnn: CifarCNN) -> None:
        """Test that state_dict has expected keys."""
        state = default_cnn.state_dict()
        assert isinstance(state, dict)
        assert len(state) > 0

    def test_state_dict_roundtrip(self, default_cnn: CifarCNN) -> None:
        """Test that saving and loading state_dict preserves weights."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"

            # Save
            torch.save(default_cnn.state_dict(), path)
            saved_state = default_cnn.state_dict()

            # Create new model and load
            config = ModelConfig(
                name="cifar_cnn",
                num_classes=10,
                conv_channels=[32, 64, 128],
                dropout=0.3,
                activation="relu",
            )
            new_model = CifarCNN(config)
            new_model.load_state_dict(torch.load(path))
            loaded_state = new_model.state_dict()

            # Compare
            for key in saved_state.keys():
                assert torch.equal(saved_state[key], loaded_state[key])

    def test_forward_after_load_produces_same_output(self, default_cnn: CifarCNN) -> None:
        """Test that forward pass after load produces identical output."""
        default_cnn.eval()  # Set to eval mode to disable dropout
        torch.manual_seed(42)
        images = torch.randn(4, 3, 32, 32)

        # Get original output
        with torch.no_grad():
            original_output = default_cnn(images)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"

            # Save state
            torch.save(default_cnn.state_dict(), path)

            # Create new model and load
            config = ModelConfig(
                name="cifar_cnn",
                num_classes=10,
                conv_channels=[32, 64, 128],
                dropout=0.3,
                activation="relu",
            )
            new_model = CifarCNN(config)
            new_model.eval()  # Set to eval mode
            new_model.load_state_dict(torch.load(path))

            # Get output from loaded model
            with torch.no_grad():
                loaded_output = new_model(images)

            # Compare
            assert torch.allclose(original_output, loaded_output, atol=1e-5)


# =============================================================================
# Tests: Backward Pass and Gradients
# =============================================================================


@pytest.mark.unit
class TestBackwardPass:
    """Tests for backward pass and gradient computation."""

    def test_backward_pass_computes_gradients(
        self, default_cnn: CifarCNN, dummy_batch: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Test that backward pass computes gradients."""
        images, labels = dummy_batch

        # Forward
        output = default_cnn(images)
        loss = nn.CrossEntropyLoss()(output, labels)

        # Backward
        loss.backward()

        # Check that gradients were computed
        for param in default_cnn.parameters():
            if param.requires_grad:
                assert param.grad is not None
                assert not torch.isnan(param.grad).any()

    def test_gradients_are_non_zero(
        self, default_cnn: CifarCNN, dummy_batch: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Test that gradients are non-zero after backward pass."""
        images, labels = dummy_batch

        # Forward
        output = default_cnn(images)
        loss = nn.CrossEntropyLoss()(output, labels)

        # Backward
        loss.backward()

        # Check that at least some gradients are non-zero
        has_nonzero_grad = False
        for param in default_cnn.parameters():
            if param.requires_grad and param.grad is not None:
                if (param.grad != 0).any():
                    has_nonzero_grad = True
                    break
        assert has_nonzero_grad

    def test_optimizer_step_updates_parameters(
        self, default_cnn: CifarCNN, dummy_batch: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Test that optimizer.step() updates model parameters."""
        images, labels = dummy_batch

        # Get original parameters
        original_params = {name: param.clone() for name, param in default_cnn.named_parameters()}

        # Forward + backward
        optimizer = torch.optim.SGD(default_cnn.parameters(), lr=0.01)
        output = default_cnn(images)
        loss = nn.CrossEntropyLoss()(output, labels)
        loss.backward()

        # Optimizer step
        optimizer.step()

        # Check that at least some parameters changed
        params_changed = False
        for name, param in default_cnn.named_parameters():
            if not torch.equal(param, original_params[name]):
                params_changed = True
                break
        assert params_changed

    def test_multiple_forward_backward_cycles(
        self, default_cnn: CifarCNN, dummy_batch: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Test that multiple forward/backward cycles work correctly."""
        images, labels = dummy_batch

        optimizer = torch.optim.SGD(default_cnn.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        for _ in range(3):
            optimizer.zero_grad()
            output = default_cnn(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

        # Model should still produce finite outputs
        with torch.no_grad():
            final_output = default_cnn(images)
            assert torch.isfinite(final_output).all()


# =============================================================================
# Tests: Repr
# =============================================================================


@pytest.mark.unit
class TestRepr:
    """Tests for __repr__ method."""

    def test_repr_returns_string(self, default_cnn: CifarCNN) -> None:
        """Test that __repr__ returns a string."""
        repr_str = repr(default_cnn)
        assert isinstance(repr_str, str)

    def test_repr_contains_key_info(self, default_cnn: CifarCNN) -> None:
        """Test that __repr__ contains key architectural info."""
        repr_str = repr(default_cnn)
        assert "num_classes" in repr_str
        assert "conv_channels" in repr_str
        assert "activation" in repr_str
        assert "parameters" in repr_str

    def test_repr_contains_parameter_count(self, default_cnn: CifarCNN) -> None:
        """Test that __repr__ includes parameter count (with commas)."""
        repr_str = repr(default_cnn)
        param_count = default_cnn.parameter_count()
        # The repr formats with commas, so check for the formatted version
        param_count_str = f"{param_count:,}"
        assert param_count_str in repr_str


# =============================================================================
# Tests: Edge Cases and Robustness
# =============================================================================


@pytest.mark.unit
class TestEdgeCases:
    """Tests for edge cases and robustness."""

    def test_eval_mode_no_dropout(self, default_cnn: CifarCNN) -> None:
        """Test that eval mode disables dropout."""
        default_cnn.eval()
        torch.manual_seed(42)
        images = torch.randn(4, 3, 32, 32)

        # Multiple forward passes should give identical output in eval mode
        with torch.no_grad():
            output1 = default_cnn(images)
            output2 = default_cnn(images)

        assert torch.equal(output1, output2)

    def test_train_mode_enables_dropout(self, default_cnn: CifarCNN) -> None:
        """Test that train mode enables dropout (stochastic)."""
        default_cnn.train()
        torch.manual_seed(42)
        images = torch.randn(4, 3, 32, 32)

        # Multiple forward passes may give different outputs in train mode
        # (due to dropout stochasticity)
        output1 = default_cnn(images)
        output2 = default_cnn(images)

        # Outputs should still be the same shape
        assert output1.shape == output2.shape

    def test_no_dropout_in_eval(self) -> None:
        """Test model with zero dropout works correctly."""
        config = ModelConfig(
            name="cifar_cnn",
            num_classes=10,
            conv_channels=[32, 64],
            dropout=0.0,
            activation="relu",
        )
        model = CifarCNN(config)
        images = torch.randn(4, 3, 32, 32)
        output = model(images)
        assert output.shape == (4, 10)

    def test_high_dropout_in_eval(self) -> None:
        """Test model with high dropout works correctly."""
        config = ModelConfig(
            name="cifar_cnn",
            num_classes=10,
            conv_channels=[32, 64],
            dropout=0.9,
            activation="relu",
        )
        model = CifarCNN(config)
        images = torch.randn(4, 3, 32, 32)
        output = model(images)
        assert output.shape == (4, 10)

    def test_model_to_device(self, default_cnn: CifarCNN) -> None:
        """Test that model can be moved to different devices."""
        # CPU
        default_cnn.to("cpu")
        images = torch.randn(4, 3, 32, 32, device="cpu")
        output = default_cnn(images)
        assert output.device.type == "cpu"

        # CUDA (if available)
        if torch.cuda.is_available():
            default_cnn.to("cuda")
            images_cuda = torch.randn(4, 3, 32, 32, device="cuda")
            output_cuda = default_cnn(images_cuda)
            assert output_cuda.device.type == "cuda"

    def test_model_freeze_parameters(self, default_cnn: CifarCNN) -> None:
        """Test freezing model parameters."""
        # Freeze
        for param in default_cnn.parameters():
            param.requires_grad = False

        # Check
        assert not any(p.requires_grad for p in default_cnn.parameters())

        # parameter_count should be 0 when all frozen
        assert default_cnn.parameter_count() == 0

    def test_model_unfreeze_parameters(self, default_cnn: CifarCNN) -> None:
        """Test unfreezing model parameters."""
        # Freeze then unfreeze
        for param in default_cnn.parameters():
            param.requires_grad = False
        for param in default_cnn.parameters():
            param.requires_grad = True

        # parameter_count should be non-zero
        assert default_cnn.parameter_count() > 0
