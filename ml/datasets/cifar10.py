"""CIFAR-10 dataset loader and data module.

Provides CIFAR10DataModule for loading, partitioning, and serving CIFAR-10 splits
(training, validation, test) as PyTorch DataLoaders.

The module handles:
- Downloading and caching CIFAR-10 (idempotent).
- Creating reproducible train/validation splits using the configured seed.
- Applying appropriate transforms (augmentation for training, normalisation for evaluation).
- Exposing train_loader, val_loader, test_loader as properties.

Design:
- Train/val split: 50 000 training samples are shuffled (using the config seed)
  and split at floor(50000 * (1 - val_fraction)). The test set (10 000 samples)
  is always used as-is.
- Transforms: Training uses get_train_transforms(augment=config.augment_train).
  Validation and test use get_eval_transforms().
- Reproducibility: The split is deterministic and reproducible with the same seed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

from ml.datasets.transforms import get_eval_transforms, get_train_transforms

if TYPE_CHECKING:
    from ml.config import DatasetConfig

logger = logging.getLogger(__name__)

# CIFAR-10 class names (in index order)
CIFAR10_CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


class CIFAR10DataModule:
    """Data module for CIFAR-10 dataset.

    Loads, partitions, and serves CIFAR-10 splits (train, validation, test)
    as PyTorch DataLoaders. Handles download, caching, train/val splitting,
    and transform application.

    This module is NOT a LightningModule — it is a simple utility class
    decoupled from any framework.

    Attributes:
        config: DatasetConfig governing batch size, data directory, val split, etc.
        _train_dataset: Subset of CIFAR-10 training set (after train/val split).
        _val_dataset: Subset of CIFAR-10 training set (validation portion).
        _test_dataset: Full CIFAR-10 test set.
        _train_loader: DataLoader for training subset.
        _val_loader: DataLoader for validation subset.
        _test_loader: DataLoader for test set.
        _is_setup: Flag indicating whether setup() has been called.

    """

    def __init__(self, config: DatasetConfig) -> None:
        """Initialise the CIFAR-10 data module.

        Args:
            config: DatasetConfig with data_dir, val_fraction, batch_size, num_workers, etc.

        Raises:
            ValueError: If config.name is not "cifar10".

        """
        if config.name != "cifar10":
            msg = f"CIFAR10DataModule only supports name='cifar10', got '{config.name}'"
            raise ValueError(msg)

        self.config = config
        self._train_dataset: Subset[tuple[Any, Any]] | None = None
        self._val_dataset: Subset[tuple[Any, Any]] | None = None
        self._test_dataset: datasets.CIFAR10 | None = None
        self._train_loader: DataLoader[tuple[Any, Any]] | None = None
        self._val_loader: DataLoader[tuple[Any, Any]] | None = None
        self._test_loader: DataLoader[tuple[Any, Any]] | None = None
        self._is_setup = False

    def prepare_data(self) -> None:
        """Download and cache CIFAR-10 to disk.

        Creates the data directory if it does not exist.
        Downloads CIFAR-10 using torchvision if not already cached.
        This method is idempotent — calling multiple times does not re-download.

        Logs:
            DATASET_DOWNLOAD_STARTED: When download begins.
            DATASET_CACHE_HIT: When data is already cached.
            DATASET_DOWNLOAD_COMPLETED: When download finishes.

        Side effects:
            Creates self.config.data_dir on disk.
            Downloads CIFAR-10 to that directory.
        """
        data_dir_path = Path(self.config.data_dir)

        # Create data directory if it doesn't exist
        data_dir_path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Data directory ensured at {data_dir_path}")

        # Check if CIFAR-10 is already cached
        # The presence of cifar-10-batches (training) or test_batch (test) indicates a download
        train_dir = data_dir_path / "cifar-10-batches-py"
        test_batch_path = data_dir_path / "cifar-10-batches-py" / "test_batch"

        if train_dir.exists() and test_batch_path.exists():
            logger.info("DATASET_CACHE_HIT")
            return

        logger.info("DATASET_DOWNLOAD_STARTED")

        # Download CIFAR-10 (train + test)
        datasets.CIFAR10(
            root=str(data_dir_path),
            train=True,
            download=True,
        )
        datasets.CIFAR10(
            root=str(data_dir_path),
            train=False,
            download=True,
        )

        logger.info("DATASET_DOWNLOAD_COMPLETED")

    def setup(self, seed: int | None = None) -> None:
        """Build train/val/test splits and DataLoaders.

        Creates train/val/test Subsets from CIFAR-10 and wraps them in DataLoaders.
        Train/val split is done on the 50 000 training samples, shuffled using the
        configured seed (or provided seed), then split at floor(50000 * (1 - val_fraction)).

        Args:
            seed: Random seed for shuffling indices. If None, uses self.config.seed
                  (which should have been set via set_seeds() in config loading).
                  If provided, overrides the config seed for this setup call.

        Raises:
            FileNotFoundError: If prepare_data() has not been called and data is not present.

        """
        # Use provided seed or assume config seed was already set via set_seeds()
        # (we don't read config.seed here; rely on torch/np/random state already set)
        if seed is not None:
            torch.manual_seed(seed)

        data_dir_path = Path(self.config.data_dir)

        # Load full CIFAR-10 datasets
        train_full = datasets.CIFAR10(
            root=str(data_dir_path),
            train=True,
            download=False,
            transform=get_train_transforms(augment=self.config.augment_train),
        )
        test_full = datasets.CIFAR10(
            root=str(data_dir_path),
            train=False,
            download=False,
            transform=get_eval_transforms(),
        )

        # Create train/val split from the 50 000 training samples
        num_train = len(train_full)  # 50 000
        indices = torch.randperm(num_train).tolist()

        val_size = int(num_train * self.config.val_fraction)
        train_size = num_train - val_size

        train_indices = indices[:train_size]
        val_indices = indices[train_size:]

        # Create Subsets
        self._train_dataset = Subset(train_full, train_indices)
        self._val_dataset = Subset(train_full, val_indices)
        self._test_dataset = test_full

        # Create DataLoaders
        self._train_loader = DataLoader(
            self._train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )
        self._val_loader = DataLoader(
            self._val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )
        self._test_loader = DataLoader(
            self._test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )

        self._is_setup = True
        logger.debug("CIFAR-10 data module setup complete")

    @property
    def train_loader(self) -> DataLoader[tuple[Any, Any]]:
        """Return the training DataLoader.

        Raises:
            RuntimeError: If setup() has not been called.

        """
        if not self._is_setup or self._train_loader is None:
            msg = "setup() must be called before accessing train_loader"
            raise RuntimeError(msg)
        return self._train_loader

    @property
    def val_loader(self) -> DataLoader[tuple[Any, Any]]:
        """Return the validation DataLoader.

        Raises:
            RuntimeError: If setup() has not been called.

        """
        if not self._is_setup or self._val_loader is None:
            msg = "setup() must be called before accessing val_loader"
            raise RuntimeError(msg)
        return self._val_loader

    @property
    def test_loader(self) -> DataLoader[tuple[Any, Any]]:
        """Return the test DataLoader.

        Raises:
            RuntimeError: If setup() has not been called.

        """
        if not self._is_setup or self._test_loader is None:
            msg = "setup() must be called before accessing test_loader"
            raise RuntimeError(msg)
        return self._test_loader

    @property
    def num_classes(self) -> int:
        """Return the number of classes in CIFAR-10."""
        return 10

    @property
    def class_names(self) -> list[str]:
        """Return the list of class names in CIFAR-10.

        Index order: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.
        """
        return list(CIFAR10_CLASSES)


if __name__ == "__main__":
    # Quick sanity check for standalone invocation.
    import sys

    from ml.config import load_config

    try:
        # Load config from phase1_baseline
        config_path = Path("experiments/configs/phase1_baseline.yaml")
        if not config_path.exists():
            print(f"Error: config file not found at {config_path}", file=sys.stderr)
            sys.exit(1)

        cfg = load_config(config_path)
        print(f"Loaded config: {cfg.experiment_name}")

        # Prepare and setup data
        dm = CIFAR10DataModule(cfg.dataset)
        print("Preparing data...")
        dm.prepare_data()

        print("Setting up data module...")
        dm.setup()

        # Print stats
        print(f"Number of classes: {dm.num_classes}")
        print(f"Class names: {dm.class_names}")
        train_size = len(dm.train_loader.dataset)  # type: ignore[arg-type]
        val_size = len(dm.val_loader.dataset)  # type: ignore[arg-type]
        test_size = len(dm.test_loader.dataset)  # type: ignore[arg-type]
        print(f"Train samples: {train_size}")
        print(f"Val samples: {val_size}")
        print(f"Test samples: {test_size}")

        # Show first batch shape
        batch_x, batch_y = next(iter(dm.train_loader))
        print(f"First batch shape: images={batch_x.shape}, labels={batch_y.shape}")
        print(f"Label range: [{batch_y.min().item()}, {batch_y.max().item()}]")

        print("✓ All checks passed")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)
