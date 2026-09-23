"""Shared pytest fixtures for FedErase tests.

This module provides reusable fixtures used across all test modules.
Fixtures here avoid real network calls and real CIFAR-10 downloads —
those are reserved for tests marked @pytest.mark.integration.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# Random seed fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def fixed_seed() -> int:
    """Set all random seeds to a fixed value and return it.

    Use this fixture in any test that requires deterministic behaviour.
    """
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


# ---------------------------------------------------------------------------
# Synthetic dataset fixtures
# These replace real CIFAR-10 in unit tests so no download is needed.
# ---------------------------------------------------------------------------

NUM_CLASSES = 10
IMAGE_CHANNELS = 3
IMAGE_SIZE = 32


@pytest.fixture
def synthetic_batch() -> tuple[torch.Tensor, torch.Tensor]:
    """Return a single synthetic batch: (images, labels).

    images shape: (8, 3, 32, 32), float32 in [-1, 1]
    labels shape: (8,), int64 in [0, 9]
    """
    torch.manual_seed(0)
    images = torch.randn(8, IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
    labels = torch.randint(0, NUM_CLASSES, (8,))
    return images, labels


@pytest.fixture
def synthetic_dataset(fixed_seed: int) -> TensorDataset:
    """Return a small synthetic TensorDataset mimicking CIFAR-10 batches.

    100 samples, shape (3, 32, 32), labels in [0, 9].
    No network access required.
    """
    n = 100
    images = torch.randn(n, IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
    labels = torch.randint(0, NUM_CLASSES, (n,))
    return TensorDataset(images, labels)


@pytest.fixture
def synthetic_train_loader(synthetic_dataset: TensorDataset) -> DataLoader:
    """DataLoader wrapping the synthetic training dataset. Batch size 16."""
    return DataLoader(synthetic_dataset, batch_size=16, shuffle=False)


@pytest.fixture
def synthetic_val_loader(synthetic_dataset: TensorDataset) -> DataLoader:
    """DataLoader wrapping a small synthetic validation dataset. Batch size 16."""
    n = 40
    images = torch.randn(n, IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
    labels = torch.randint(0, NUM_CLASSES, (n,))
    val_ds = TensorDataset(images, labels)
    return DataLoader(val_ds, batch_size=16, shuffle=False)


@pytest.fixture
def tiny_train_loader() -> DataLoader:
    """Single-batch DataLoader with 4 samples — for fast smoke tests."""
    images = torch.randn(4, IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
    labels = torch.randint(0, NUM_CLASSES, (4,))
    ds = TensorDataset(images, labels)
    return DataLoader(ds, batch_size=4, shuffle=False)


@pytest.fixture
def tiny_val_loader() -> DataLoader:
    """Single-batch validation DataLoader with 4 samples."""
    images = torch.randn(4, IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
    labels = torch.randint(0, NUM_CLASSES, (4,))
    ds = TensorDataset(images, labels)
    return DataLoader(ds, batch_size=4, shuffle=False)
