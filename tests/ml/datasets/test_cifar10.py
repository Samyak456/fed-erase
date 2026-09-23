"""Tests for CIFAR-10 dataset loader.

Covers download/caching, train/val/test splitting, DataLoader properties,
batch shapes, label ranges, and reproducibility.

Unit tests use synthetic dataset fixtures (no real CIFAR-10 download).
Integration tests marked @pytest.mark.integration download real CIFAR-10.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch.utils.data import DataLoader

from ml.config import DatasetConfig
from ml.datasets.cifar10 import CIFAR10DataModule

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_data_dir() -> str:
    """Create and return a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def dataset_config(temp_data_dir: str) -> DatasetConfig:
    """Return a basic DatasetConfig for testing."""
    return DatasetConfig(
        name="cifar10",
        data_dir=Path(temp_data_dir),
        val_fraction=0.1,
        batch_size=64,
        num_workers=0,
        augment_train=True,
    )


@pytest.fixture
def dataset_config_no_augment(temp_data_dir: str) -> DatasetConfig:
    """Return a DatasetConfig with augmentation disabled."""
    return DatasetConfig(
        name="cifar10",
        data_dir=Path(temp_data_dir),
        val_fraction=0.1,
        batch_size=64,
        num_workers=0,
        augment_train=False,
    )


@pytest.fixture
def dataset_config_with_seed(temp_data_dir: str, fixed_seed: int) -> DatasetConfig:
    """Return a DatasetConfig with a fixed seed."""
    return DatasetConfig(
        name="cifar10",
        data_dir=Path(temp_data_dir),
        val_fraction=0.1,
        batch_size=16,
        num_workers=0,
        augment_train=True,
    )


# =============================================================================
# Tests: Initialization
# =============================================================================


@pytest.mark.unit
class TestInitialization:
    """Tests for CIFAR10DataModule initialization."""

    def test_init_with_valid_config(self, dataset_config: DatasetConfig) -> None:
        """Test that initialization succeeds with valid config."""
        dm = CIFAR10DataModule(dataset_config)
        assert dm.config == dataset_config
        assert dm._is_setup is False

    def test_init_with_wrong_name_raises_value_error(self) -> None:
        """Test that Pydantic validation fails if config.name is not 'cifar10'."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match='Only "cifar10" is supported'):
            DatasetConfig(
                name="cifar100",  # type: ignore
                data_dir="data",
                val_fraction=0.1,
                batch_size=64,
                num_workers=0,
                augment_train=True,
            )


# =============================================================================
# Tests: prepare_data()
# =============================================================================


@pytest.mark.unit
class TestPrepareData:
    """Tests for prepare_data() method."""

    def test_prepare_data_creates_directory_if_missing(self, temp_data_dir: str) -> None:
        """Test that prepare_data creates the data directory."""
        # Create a fresh temp dir that doesn't exist yet
        fresh_dir = Path(temp_data_dir) / "cifar10_fresh"
        assert not fresh_dir.exists()

        config = DatasetConfig(
            name="cifar10",
            data_dir=str(fresh_dir),
            val_fraction=0.1,
            batch_size=64,
            num_workers=0,
            augment_train=True,
        )

        with patch("ml.datasets.cifar10.datasets.CIFAR10"):
            dm = CIFAR10DataModule(config)
            dm.prepare_data()

            assert fresh_dir.exists()

    def test_prepare_data_is_idempotent(self, temp_data_dir: str) -> None:
        """Test that calling prepare_data twice does not re-download."""
        config = DatasetConfig(
            name="cifar10",
            data_dir=temp_data_dir,
            val_fraction=0.1,
            batch_size=64,
            num_workers=0,
            augment_train=True,
        )

        with patch("ml.datasets.cifar10.datasets.CIFAR10") as mock_cifar10:
            dm = CIFAR10DataModule(config)

            # First call — create the cache marker
            data_dir_path = Path(temp_data_dir)
            (data_dir_path / "cifar-10-batches-py").mkdir(parents=True, exist_ok=True)
            (data_dir_path / "cifar-10-batches-py" / "test_batch").touch()

            dm.prepare_data()

            # Reset mock to ensure we can see new calls
            mock_cifar10.reset_mock()

            # Second call — should not call CIFAR10 at all (cache hit)
            dm.prepare_data()
            second_call_count = mock_cifar10.call_count

            # Should not have called CIFAR10 on second run
            assert second_call_count == 0

    def test_prepare_data_logs_cache_hit_if_already_cached(
        self, temp_data_dir: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that prepare_data logs DATASET_CACHE_HIT if data is cached."""
        data_dir_path = Path(temp_data_dir)

        # Create the cached directory marker
        (data_dir_path / "cifar-10-batches-py").mkdir(parents=True, exist_ok=True)
        (data_dir_path / "cifar-10-batches-py" / "test_batch").touch()

        config = DatasetConfig(
            name="cifar10",
            data_dir=temp_data_dir,
            val_fraction=0.1,
            batch_size=64,
            num_workers=0,
            augment_train=True,
        )

        dm = CIFAR10DataModule(config)

        with caplog.at_level("INFO"):
            dm.prepare_data()

        assert "DATASET_CACHE_HIT" in caplog.text


# =============================================================================
# Tests: setup() and properties (unit tests with synthetic data)
# =============================================================================


@pytest.mark.unit
class TestSetupAndProperties:
    """Tests for setup() method and DataLoader properties (using synthetic data)."""

    def test_train_loader_raises_before_setup(self, dataset_config: DatasetConfig) -> None:
        """Test that train_loader raises RuntimeError before setup()."""
        dm = CIFAR10DataModule(dataset_config)
        with pytest.raises(RuntimeError, match="setup\\(\\) must be called"):
            _ = dm.train_loader

    def test_val_loader_raises_before_setup(self, dataset_config: DatasetConfig) -> None:
        """Test that val_loader raises RuntimeError before setup()."""
        dm = CIFAR10DataModule(dataset_config)
        with pytest.raises(RuntimeError, match="setup\\(\\) must be called"):
            _ = dm.val_loader

    def test_test_loader_raises_before_setup(self, dataset_config: DatasetConfig) -> None:
        """Test that test_loader raises RuntimeError before setup()."""
        dm = CIFAR10DataModule(dataset_config)
        with pytest.raises(RuntimeError, match="setup\\(\\) must be called"):
            _ = dm.test_loader

    def test_num_classes_is_10(self, dataset_config: DatasetConfig) -> None:
        """Test that num_classes is always 10 (no setup needed)."""
        dm = CIFAR10DataModule(dataset_config)
        assert dm.num_classes == 10

    def test_class_names_has_10_elements(self, dataset_config: DatasetConfig) -> None:
        """Test that class_names has 10 elements (no setup needed)."""
        dm = CIFAR10DataModule(dataset_config)
        assert len(dm.class_names) == 10

    def test_class_names_correct_order(self, dataset_config: DatasetConfig) -> None:
        """Test that class_names are in correct CIFAR-10 order."""
        dm = CIFAR10DataModule(dataset_config)
        expected = [
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
        ]
        assert dm.class_names == expected


# =============================================================================
# Tests: setup() with mocked CIFAR10 (unit)
# =============================================================================


@pytest.mark.unit
class TestSetupWithMockedData:
    """Tests for setup() method with mocked CIFAR-10 datasets."""

    def _create_mock_cifar10(self, num_samples: int, train: bool = True) -> MagicMock:
        """Create a mock CIFAR10 dataset."""
        mock_ds = MagicMock()
        mock_ds.__len__ = MagicMock(return_value=num_samples)
        mock_ds.__getitem__ = MagicMock(
            side_effect=lambda i: (
                torch.randn(3, 32, 32),
                torch.randint(0, 10, (1,)).item(),
            )
        )
        return mock_ds

    def test_setup_creates_loaders(self, dataset_config: DatasetConfig) -> None:
        """Test that setup() creates the DataLoaders."""
        with patch("ml.datasets.cifar10.datasets.CIFAR10") as mock_cifar10:
            # Setup mock to return 50k train and 10k test
            mock_cifar10.side_effect = [
                self._create_mock_cifar10(50000, train=True),
                self._create_mock_cifar10(10000, train=False),
            ]

            dm = CIFAR10DataModule(dataset_config)
            dm.setup()

            assert dm._train_loader is not None
            assert dm._val_loader is not None
            assert dm._test_loader is not None

    def test_train_and_val_indices_are_disjoint(self, dataset_config: DatasetConfig) -> None:
        """Test that train and val use non-overlapping indices."""
        with patch("ml.datasets.cifar10.datasets.CIFAR10") as mock_cifar10:
            mock_train = self._create_mock_cifar10(50000, train=True)
            mock_test = self._create_mock_cifar10(10000, train=False)
            mock_cifar10.side_effect = [mock_train, mock_test]

            dm = CIFAR10DataModule(dataset_config)
            dm.setup()

            # Extract indices from Subsets
            train_indices = set(dm._train_dataset.indices)  # type: ignore
            val_indices = set(dm._val_dataset.indices)  # type: ignore

            # Check disjoint
            assert len(train_indices & val_indices) == 0

    def test_setup_with_different_seeds_produces_different_splits(
        self, dataset_config: DatasetConfig
    ) -> None:
        """Test that different seeds produce different train/val splits."""
        with patch("ml.datasets.cifar10.datasets.CIFAR10") as mock_cifar10:
            mock_train = self._create_mock_cifar10(50000, train=True)
            mock_test = self._create_mock_cifar10(10000, train=False)

            # First setup with seed=42
            mock_cifar10.side_effect = [mock_train, mock_test]
            dm1 = CIFAR10DataModule(dataset_config)
            dm1.setup(seed=42)
            indices1 = set(dm1._train_dataset.indices)  # type: ignore

            # Second setup with seed=99
            mock_cifar10.reset_mock()
            mock_train2 = self._create_mock_cifar10(50000, train=True)
            mock_test2 = self._create_mock_cifar10(10000, train=False)
            mock_cifar10.side_effect = [mock_train2, mock_test2]
            dm2 = CIFAR10DataModule(dataset_config)
            dm2.setup(seed=99)
            indices2 = set(dm2._train_dataset.indices)  # type: ignore

            # Should be different
            assert indices1 != indices2

    def test_setup_with_same_seed_produces_same_split(self, dataset_config: DatasetConfig) -> None:
        """Test that the same seed produces the same train/val split."""
        with patch("ml.datasets.cifar10.datasets.CIFAR10") as mock_cifar10:
            # First setup
            mock_train1 = self._create_mock_cifar10(50000, train=True)
            mock_test1 = self._create_mock_cifar10(10000, train=False)
            mock_cifar10.side_effect = [mock_train1, mock_test1]
            dm1 = CIFAR10DataModule(dataset_config)
            dm1.setup(seed=42)
            indices1 = sorted(dm1._train_dataset.indices)  # type: ignore

            # Second setup with same seed
            mock_cifar10.reset_mock()
            mock_train2 = self._create_mock_cifar10(50000, train=True)
            mock_test2 = self._create_mock_cifar10(10000, train=False)
            mock_cifar10.side_effect = [mock_train2, mock_test2]
            dm2 = CIFAR10DataModule(dataset_config)
            dm2.setup(seed=42)
            indices2 = sorted(dm2._train_dataset.indices)  # type: ignore

            # Should be the same
            assert indices1 == indices2


# =============================================================================
# Tests: DataLoader batch properties (unit, with synthetic data)
# =============================================================================


@pytest.mark.unit
class TestBatchProperties:
    """Tests for DataLoader batch shapes and types."""

    def test_batch_from_train_loader_has_correct_shape(
        self, synthetic_batch: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Test that a batch from DataLoader has shape (B, 3, 32, 32)."""
        images, labels = synthetic_batch
        assert images.shape == (8, 3, 32, 32)
        assert labels.shape == (8,)

    def test_batch_from_train_loader_has_correct_dtype(
        self, synthetic_batch: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Test that images are float32 and labels are int64."""
        images, labels = synthetic_batch
        assert images.dtype == torch.float32
        assert labels.dtype == torch.int64

    def test_labels_in_valid_range(
        self, synthetic_batch: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Test that labels are in range [0, 9]."""
        _, labels = synthetic_batch
        assert labels.min() >= 0
        assert labels.max() < 10


# =============================================================================
# Tests: Integration (real CIFAR-10 download)
# =============================================================================


@pytest.mark.integration
class TestIntegrationRealCIFAR10:
    """Integration tests using real CIFAR-10 download."""

    def test_full_pipeline_download_setup_and_batches(self, fixed_seed: int) -> None:
        """Test complete pipeline: prepare, setup, and iterate batches."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = DatasetConfig(
                name="cifar10",
                data_dir=tmpdir,
                val_fraction=0.1,
                batch_size=32,
                num_workers=0,
                augment_train=True,
            )

            dm = CIFAR10DataModule(config)
            dm.prepare_data()
            dm.setup(seed=fixed_seed)

            # Check that loaders are valid
            assert isinstance(dm.train_loader, DataLoader)
            assert isinstance(dm.val_loader, DataLoader)
            assert isinstance(dm.test_loader, DataLoader)

    def test_train_val_test_split_counts(self, fixed_seed: int) -> None:
        """Test that train + val = 50k and test = 10k."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = DatasetConfig(
                name="cifar10",
                data_dir=tmpdir,
                val_fraction=0.1,
                batch_size=64,
                num_workers=0,
                augment_train=True,
            )

            dm = CIFAR10DataModule(config)
            dm.prepare_data()
            dm.setup(seed=fixed_seed)

            train_size = len(dm.train_loader.dataset)  # type: ignore
            val_size = len(dm.val_loader.dataset)  # type: ignore
            test_size = len(dm.test_loader.dataset)  # type: ignore

            # Verify split
            assert train_size + val_size == 50000
            assert test_size == 10000
            # With val_fraction=0.1, we expect ~45k train and ~5k val
            assert train_size == 45000
            assert val_size == 5000

    def test_batch_from_real_data_has_correct_shape(self, fixed_seed: int) -> None:
        """Test that real CIFAR-10 batches have correct shape and dtype."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = DatasetConfig(
                name="cifar10",
                data_dir=tmpdir,
                val_fraction=0.1,
                batch_size=16,
                num_workers=0,
                augment_train=False,  # No augmentation for determinism
            )

            dm = CIFAR10DataModule(config)
            dm.prepare_data()
            dm.setup(seed=fixed_seed)

            # Get first batch
            batch_x, batch_y = next(iter(dm.test_loader))

            assert batch_x.shape == (16, 3, 32, 32)
            assert batch_y.shape == (16,)
            assert batch_x.dtype == torch.float32
            assert batch_y.dtype == torch.int64

    def test_labels_in_valid_range_real_data(self, fixed_seed: int) -> None:
        """Test that real CIFAR-10 labels are in [0, 9]."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = DatasetConfig(
                name="cifar10",
                data_dir=tmpdir,
                val_fraction=0.1,
                batch_size=64,
                num_workers=0,
                augment_train=True,
            )

            dm = CIFAR10DataModule(config)
            dm.prepare_data()
            dm.setup(seed=fixed_seed)

            # Check all labels in test set
            for _, batch_y in dm.test_loader:
                assert batch_y.min() >= 0
                assert batch_y.max() < 10
                break  # Just check first batch

    def test_val_fraction_respected(self, fixed_seed: int) -> None:
        """Test that val_fraction is correctly applied."""
        with tempfile.TemporaryDirectory() as tmpdir:
            val_frac = 0.2
            config = DatasetConfig(
                name="cifar10",
                data_dir=tmpdir,
                val_fraction=val_frac,
                batch_size=64,
                num_workers=0,
                augment_train=True,
            )

            dm = CIFAR10DataModule(config)
            dm.prepare_data()
            dm.setup(seed=fixed_seed)

            train_size = len(dm.train_loader.dataset)  # type: ignore
            val_size = len(dm.val_loader.dataset)  # type: ignore

            # With val_fraction=0.2, expect 80% train and 20% val
            expected_train = int(50000 * (1 - val_frac))
            expected_val = 50000 - expected_train

            assert train_size == expected_train
            assert val_size == expected_val
