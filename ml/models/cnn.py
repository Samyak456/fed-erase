"""CIFAR-10 CNN architecture.

Provides CifarCNN, a configurable convolutional neural network for CIFAR-10 image
classification. The network consists of:

1. Dynamic convolutional blocks (Conv2d → BatchNorm → Activation → MaxPool)
   controlled by ModelConfig.conv_channels
2. Flattening layer
3. Two fully connected layers with dropout

The architecture does NOT apply softmax — outputs raw logits for use with
nn.CrossEntropyLoss.

Design decisions:
- Conv layers use 3x3 kernels with padding=1 (preserve spatial dims per block).
- Each block halves spatial dimensions via MaxPool(2x2).
- FC input dimensions are computed dynamically via a dummy forward pass,
  not hard-coded, to avoid errors when changing conv_channels.
- Dropout is applied before the final FC layer.
- Activation function (relu or gelu) is configurable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from ml.config import ModelConfig
else:
    ModelConfig = None


class CifarCNN(nn.Module):
    """Configurable CNN for CIFAR-10 classification.

    Architecture:
    - Dynamic conv blocks: [Conv2d(k_in -> k_out, 3x3, pad=1) -> BN -> Activation -> MaxPool(2x2)]
    - Flatten
    - FC(feature_dim -> 256) -> Activation -> Dropout
    - FC(256 -> num_classes)

    The number of conv blocks and filter counts are driven by config.conv_channels.
    For example:
    - conv_channels=[32, 64, 128] -> 3 blocks: 3->32, 32->64, 64->128
    - conv_channels=[64, 128] -> 2 blocks: 3->64, 64->128

    Attributes:
        config: ModelConfig controlling architecture.
        conv_blocks: Sequential of convolutional blocks.
        fc_input_dim: Computed dimension after flattening final conv output.
        fc: Sequential of fully connected layers (fc_input_dim -> 256 -> num_classes).
        dropout: Dropout layer before final FC.
        activation: Activation function (relu or gelu).

    """

    def __init__(self, config: ModelConfig) -> None:
        """Initialise CifarCNN.

        Args:
            config: ModelConfig with conv_channels, activation, num_classes, dropout.

        Raises:
            ValueError: If config.activation is not "relu" or "gelu".

        """
        super().__init__()
        self.config = config

        # Validate activation
        if config.activation not in ("relu", "gelu"):
            msg = f'Unsupported activation: {config.activation}. Must be "relu" or "gelu".'
            raise ValueError(msg)

        # Build activation function
        if config.activation == "relu":
            activation_fn: nn.Module = nn.ReLU(inplace=True)
        else:
            activation_fn = nn.GELU()
        self.activation = activation_fn

        # Build convolutional blocks dynamically
        conv_blocks_list: list[nn.Module] = []
        in_channels = 3
        for out_channels in config.conv_channels:
            conv_blocks_list.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
            )
            conv_blocks_list.append(nn.BatchNorm2d(out_channels))
            conv_blocks_list.append(self.activation)
            conv_blocks_list.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = out_channels

        self.conv_blocks = nn.Sequential(*conv_blocks_list)

        # Compute FC input dimension via dummy forward pass
        # Input (1, 3, 32, 32) → after conv blocks → (1, last_channels, h, w)
        dummy_input = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            dummy_output = self.conv_blocks(dummy_input)
        self.fc_input_dim = dummy_output.numel()  # Total elements in flattened output

        # Build fully connected layers
        # fc_input_dim → 256 → num_classes
        self.dropout = nn.Dropout(p=config.dropout)
        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_dim, 256, bias=True),
            self.activation,
            nn.Linear(256, config.num_classes, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the CNN.

        Args:
            x: Input tensor of shape (B, 3, 32, 32).

        Returns:
            Output tensor of shape (B, num_classes) — raw logits (no softmax).

        """
        # Conv blocks
        x = self.conv_blocks(x)

        # Flatten
        x = x.flatten(start_dim=1)

        # Fully connected layers with dropout
        x = self.dropout(x)
        x = self.fc(x)

        return x

    def parameter_count(self) -> int:
        """Return the total number of trainable parameters.

        Returns:
            Count of all trainable parameters in the model.

        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        """Return a string representation of the model architecture."""
        param_count = self.parameter_count()
        num_layers = sum(
            1 for _ in self.conv_blocks if isinstance(_, (nn.Conv2d, nn.Linear, nn.BatchNorm2d))
        )
        return (
            f"CifarCNN(\n"
            f"  num_classes={self.config.num_classes},\n"
            f"  conv_channels={self.config.conv_channels},\n"
            f"  activation={self.config.activation},\n"
            f"  dropout={self.config.dropout},\n"
            f"  num_layers={num_layers},\n"
            f"  parameters={param_count:,}\n"
            f")"
        )



class ModelFactory:
    """Factory for creating model instances from configuration.

    Provides a unified interface for model creation, supporting
    multiple model types (currently CIFAR-10 CNN).
    """

    @staticmethod
    def create(config: ModelConfig) -> nn.Module:
        """Create a model instance from configuration.

        Args:
            config: ModelConfig specifying model type and parameters.

        Returns:
            Instantiated model (nn.Module).

        Raises:
            ValueError: If model name is not supported.

        """
        if config.name == "cifar_cnn":
            return CifarCNN(config)
        else:
            msg = f'Unknown model: {config.name}. Supported: ["cifar_cnn"]'
            raise ValueError(msg)
