"""Tests for checkpoint management.

Covers checkpoint save/load, metadata, checksums, versioning, and edge cases.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from ml.training.checkpoint import CheckpointManager, CheckpointMetadata

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_checkpoint_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for checkpoints."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def simple_model() -> nn.Module:
    """Create a simple model for checkpointing."""
    return nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 10),
    )


@pytest.fixture
def sample_metadata() -> CheckpointMetadata:
    """Create sample checkpoint metadata."""
    return CheckpointMetadata(
        model_name="simple_model",
        epoch=5,
        val_accuracy=0.95,
        val_loss=0.15,
        total_epochs_trained=5,
        config_hash="abcd1234efgh5678ijkl90mnopqrstu",
        model_architecture="[Linear(10, 32), ReLU(), Linear(32, 10)]",
        pytorch_version="2.0.0",
        checkpoint_version=1,
    )


# =============================================================================
# Tests: Initialization
# =============================================================================


@pytest.mark.unit
class TestCheckpointManagerInit:
    """Tests for CheckpointManager initialization."""

    def test_init_creates_directory(self, temp_checkpoint_dir: Path) -> None:
        """Test that initialization creates checkpoint directory if missing."""
        checkpoint_dir = temp_checkpoint_dir / "new_checkpoints"
        assert not checkpoint_dir.exists()
        manager = CheckpointManager(checkpoint_dir)
        assert checkpoint_dir.exists()
        assert manager.checkpoint_dir == checkpoint_dir

    def test_init_with_existing_directory(self, temp_checkpoint_dir: Path) -> None:
        """Test initialization with existing directory."""
        manager = CheckpointManager(temp_checkpoint_dir)
        assert manager.checkpoint_dir == temp_checkpoint_dir

    def test_init_keep_best_only_flag(
        self, temp_checkpoint_dir: Path
    ) -> None:
        """Test that keep_best_only flag is stored."""
        manager = CheckpointManager(temp_checkpoint_dir, keep_best_only=False)
        assert not manager.keep_best_only


# =============================================================================
# Tests: Save Functionality
# =============================================================================


@pytest.mark.unit
class TestCheckpointSave:
    """Tests for checkpoint saving."""

    def test_save_creates_state_dict_file(
        self,
        temp_checkpoint_dir: Path,
        simple_model: nn.Module,
        sample_metadata: CheckpointMetadata,
    ) -> None:
        """Test that save creates a .pt file."""
        manager = CheckpointManager(temp_checkpoint_dir)
        checkpoint_path = manager.save(simple_model, sample_metadata, name="test")
        assert checkpoint_path.exists()
        assert checkpoint_path.suffix == ".pt"
        assert checkpoint_path.name == "test.pt"

    def test_save_creates_metadata_file(
        self,
        temp_checkpoint_dir: Path,
        simple_model: nn.Module,
        sample_metadata: CheckpointMetadata,
    ) -> None:
        """Test that save creates a .meta.json file."""
        manager = CheckpointManager(temp_checkpoint_dir)
        checkpoint_path = manager.save(simple_model, sample_metadata, name="test")
        meta_path = checkpoint_path.parent / checkpoint_path.name.replace(".pt", ".meta.json")
        assert meta_path.exists()

    def test_save_creates_checksum_file(
        self,
        temp_checkpoint_dir: Path,
        simple_model: nn.Module,
        sample_metadata: CheckpointMetadata,
    ) -> None:
        """Test that save creates a .sha256 checksum file."""
        manager = CheckpointManager(temp_checkpoint_dir)
        checkpoint_path = manager.save(simple_model, sample_metadata, name="test")
        checksum_path = checkpoint_path.parent / checkpoint_path.name.replace(".pt", ".sha256")
        assert checksum_path.exists()

    def test_save_metadata_content(
        self,
        temp_checkpoint_dir: Path,
        simple_model: nn.Module,
        sample_metadata: CheckpointMetadata,
    ) -> None:
        """Test that saved metadata contains all fields."""
        manager = CheckpointManager(temp_checkpoint_dir)
        checkpoint_path = manager.save(simple_model, sample_metadata, name="test")
        meta_path = checkpoint_path.parent / checkpoint_path.name.replace(".pt", ".meta.json")
        with meta_path.open("r") as f:
            stored_metadata = json.load(f)
        assert stored_metadata["model_name"] == "simple_model"
        assert stored_metadata["epoch"] == 5
        assert stored_metadata["val_accuracy"] == 0.95
        assert "checksum" in stored_metadata

    def test_save_checksum_is_hex(
        self,
        temp_checkpoint_dir: Path,
        simple_model: nn.Module,
        sample_metadata: CheckpointMetadata,
    ) -> None:
        """Test that checksum is a valid 64-char hex string."""
        manager = CheckpointManager(temp_checkpoint_dir)
        checkpoint_path = manager.save(simple_model, sample_metadata, name="test")
        checksum_path = checkpoint_path.parent / checkpoint_path.name.replace(".pt", ".sha256")
        checksum = checksum_path.read_text().strip()
        assert len(checksum) == 64
        assert all(c in "0123456789abcdef" for c in checksum)

    def test_save_returns_path(
        self,
        temp_checkpoint_dir: Path,
        simple_model: nn.Module,
        sample_metadata: CheckpointMetadata,
    ) -> None:
        """Test that save returns the checkpoint path."""
        manager = CheckpointManager(temp_checkpoint_dir)
        result = manager.save(simple_model, sample_metadata, name="test")
        assert isinstance(result, Path)
        assert result.exists()

    def test_save_state_dict_format(
        self,
        temp_checkpoint_dir: Path,
        simple_model: nn.Module,
        sample_metadata: CheckpointMetadata,
    ) -> None:
        """Test that saved file is a valid PyTorch state_dict."""
        manager = CheckpointManager(temp_checkpoint_dir)
        checkpoint_path = manager.save(simple_model, sample_metadata, name="test")
        state_dict = torch.load(checkpoint_path, weights_only=True)
        assert isinstance(state_dict, dict)
        assert len(state_dict) > 0
        # Check that keys match model's state_dict keys
        assert set(state_dict.keys()) == set(simple_model.state_dict().keys())


# =============================================================================
# Tests: Load Functionality
# =============================================================================


@pytest.mark.unit
class TestCheckpointLoad:
    """Tests for checkpoint loading."""

    def test_load_missing_checkpoint_raises(
        self, temp_checkpoint_dir: Path, simple_model: nn.Module
    ) -> None:
        """Test that load raises FileNotFoundError for missing checkpoint."""
        manager = CheckpointManager(temp_checkpoint_dir)
        with pytest.raises(FileNotFoundError, match="Checkpoint file not found"):
            manager.load(simple_model, temp_checkpoint_dir / "nonexistent.pt")

    def test_load_missing_metadata_raises(
        self,
        temp_checkpoint_dir: Path,
        simple_model: nn.Module,
        sample_metadata: CheckpointMetadata,
    ) -> None:
        """Test that load raises if metadata file is missing."""
        manager = CheckpointManager(temp_checkpoint_dir)
        checkpoint_path = manager.save(simple_model, sample_metadata, name="test")

        # Delete metadata file
        meta_path = checkpoint_path.parent / checkpoint_path.name.replace(".pt", ".meta.json")
        meta_path.unlink()

        with pytest.raises(FileNotFoundError, match="Metadata file not found"):
            manager.load(simple_model, checkpoint_path)

    def test_load_missing_checksum_raises(
        self,
        temp_checkpoint_dir: Path,
        simple_model: nn.Module,
        sample_metadata: CheckpointMetadata,
    ) -> None:
        """Test that load raises if checksum file is missing."""
        manager = CheckpointManager(temp_checkpoint_dir)
        checkpoint_path = manager.save(simple_model, sample_metadata, name="test")

        # Delete checksum file
        checksum_path = checkpoint_path.parent / checkpoint_path.name.replace(".pt", ".sha256")
        checksum_path.unlink()

        with pytest.raises(FileNotFoundError, match="Checksum file not found"):
            manager.load(simple_model, checkpoint_path)

    def test_load_corrupted_checkpoint_raises(
        self,
        temp_checkpoint_dir: Path,
        simple_model: nn.Module,
        sample_metadata: CheckpointMetadata,
    ) -> None:
        """Test that load raises if checksum does not match (file corrupted)."""
        manager = CheckpointManager(temp_checkpoint_dir)
        checkpoint_path = manager.save(simple_model, sample_metadata, name="test")

        # Corrupt the checkpoint file by appending random bytes
        with checkpoint_path.open("ab") as f:
            f.write(b"corrupted")

        with pytest.raises(ValueError, match="Checksum mismatch"):
            manager.load(simple_model, checkpoint_path)

    def test_load_into_model_updates_weights(
        self,
        temp_checkpoint_dir: Path,
        simple_model: nn.Module,
        sample_metadata: CheckpointMetadata,
    ) -> None:
        """Test that load actually updates model weights."""
        manager = CheckpointManager(temp_checkpoint_dir)

        # Save model with initial weights
        original_weights = {k: v.clone() for k, v in simple_model.state_dict().items()}
        checkpoint_path = manager.save(simple_model, sample_metadata, name="test")

        # Modify model weights
        for param in simple_model.parameters():
            param.data.fill_(999.0)

        # Verify weights changed
        for k, v in simple_model.state_dict().items():
            assert not torch.allclose(v, original_weights[k])

        # Load checkpoint
        manager.load(simple_model, checkpoint_path)

        # Verify weights restored
        for k, v in simple_model.state_dict().items():
            assert torch.allclose(v, original_weights[k])

    def test_load_returns_metadata(
        self,
        temp_checkpoint_dir: Path,
        simple_model: nn.Module,
        sample_metadata: CheckpointMetadata,
    ) -> None:
        """Test that load returns CheckpointMetadata."""
        manager = CheckpointManager(temp_checkpoint_dir)
        checkpoint_path = manager.save(simple_model, sample_metadata, name="test")
        loaded_metadata = manager.load(simple_model, checkpoint_path)
        assert isinstance(loaded_metadata, CheckpointMetadata)
        assert loaded_metadata.model_name == "simple_model"
        assert loaded_metadata.epoch == 5
        assert loaded_metadata.val_accuracy == 0.95

    def test_load_string_path_works(
        self,
        temp_checkpoint_dir: Path,
        simple_model: nn.Module,
        sample_metadata: CheckpointMetadata,
    ) -> None:
        """Test that load works with string path (converts to Path)."""
        manager = CheckpointManager(temp_checkpoint_dir)
        checkpoint_path = manager.save(simple_model, sample_metadata, name="test")
        # Pass as string
        metadata = manager.load(simple_model, str(checkpoint_path))
        assert isinstance(metadata, CheckpointMetadata)


# =============================================================================
# Tests: Save-Load Round-Trip
# =============================================================================


@pytest.mark.unit
class TestCheckpointRoundTrip:
    """Tests for save → load → evaluate equivalence."""

    def test_save_load_preserves_model_output(
        self,
        temp_checkpoint_dir: Path,
        simple_model: nn.Module,
        sample_metadata: CheckpointMetadata,
    ) -> None:
        """Test that save/load preserves model output."""
        manager = CheckpointManager(temp_checkpoint_dir)

        # Create input and compute output with original model
        torch.manual_seed(42)
        test_input = torch.randn(4, 10)
        original_output = simple_model(test_input).detach()

        # Save checkpoint
        checkpoint_path = manager.save(simple_model, sample_metadata, name="test")

        # Create new model instance and load checkpoint
        new_model = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 10),
        )
        manager.load(new_model, checkpoint_path)

        # Compute output with loaded model
        loaded_output = new_model(test_input).detach()

        # Outputs should match exactly
        assert torch.allclose(original_output, loaded_output, atol=1e-6)

    def test_save_load_checksum_valid(
        self,
        temp_checkpoint_dir: Path,
        simple_model: nn.Module,
        sample_metadata: CheckpointMetadata,
    ) -> None:
        """Test that saved checkpoint checksum is valid and verifiable."""
        manager = CheckpointManager(temp_checkpoint_dir)
        checkpoint_path1 = manager.save(simple_model, sample_metadata, name="test1")
        checksum_path = checkpoint_path1.parent / checkpoint_path1.name.replace(".pt", ".sha256")
        stored_checksum = checksum_path.read_text().strip()

        # Verify checksum is valid
        computed_checksum = CheckpointManager._compute_file_checksum(checkpoint_path1)
        assert stored_checksum == computed_checksum
        assert len(stored_checksum) == 64

    def test_save_modified_model_different_hash(
        self,
        temp_checkpoint_dir: Path,
        simple_model: nn.Module,
        sample_metadata: CheckpointMetadata,
    ) -> None:
        """Test that saving modified model produces different checksum."""
        manager = CheckpointManager(temp_checkpoint_dir)
        checkpoint_path1 = manager.save(simple_model, sample_metadata, name="test1")
        checksum_path = checkpoint_path1.parent / checkpoint_path1.name.replace(".pt", ".sha256")
        checksum1 = checksum_path.read_text().strip()

        # Modify model
        for param in simple_model.parameters():
            param.data.add_(0.1)

        checkpoint_path2 = manager.save(simple_model, sample_metadata, name="test2")
        checksum_path = checkpoint_path2.parent / checkpoint_path2.name.replace(".pt", ".sha256")
        checksum2 = checksum_path.read_text().strip()

        assert checksum1 != checksum2


# =============================================================================
# Tests: Versioning and Best Model
# =============================================================================


@pytest.mark.unit
class TestCheckpointVersioning:
    """Tests for checkpoint versioning and best model tracking."""

    def test_save_same_name_twice_creates_version(
        self,
        temp_checkpoint_dir: Path,
        simple_model: nn.Module,
        sample_metadata: CheckpointMetadata,
    ) -> None:
        """Test that saving with same name twice creates versions."""
        manager = CheckpointManager(temp_checkpoint_dir)
        checkpoint_path1 = manager.save(simple_model, sample_metadata, name="best")
        assert checkpoint_path1.name == "best.pt"

        checkpoint_path2 = manager.save(simple_model, sample_metadata, name="best")
        assert checkpoint_path2.name == "best_v1.pt"

        # Both files should exist
        assert (temp_checkpoint_dir / "best.pt").exists()
        assert (temp_checkpoint_dir / "best_v1.pt").exists()

    def test_save_multiple_versions_increment(
        self,
        temp_checkpoint_dir: Path,
        simple_model: nn.Module,
        sample_metadata: CheckpointMetadata,
    ) -> None:
        """Test that multiple saves create incrementing versions."""
        manager = CheckpointManager(temp_checkpoint_dir)
        paths = []
        for _ in range(3):
            path = manager.save(simple_model, sample_metadata, name="best")
            paths.append(path)

        assert paths[0].name == "best.pt"
        assert paths[1].name == "best_v1.pt"
        assert paths[2].name == "best_v2.pt"

    def test_list_checkpoints(
        self,
        temp_checkpoint_dir: Path,
        simple_model: nn.Module,
        sample_metadata: CheckpointMetadata,
    ) -> None:
        """Test that list_checkpoints returns all checkpoint names."""
        manager = CheckpointManager(temp_checkpoint_dir)
        manager.save(simple_model, sample_metadata, name="best")
        manager.save(simple_model, sample_metadata, name="latest")

        checkpoints = manager.list_checkpoints()
        assert len(checkpoints) == 2
        assert "best" in checkpoints
        assert "latest" in checkpoints


# =============================================================================
# Tests: Get Checkpoint Path
# =============================================================================


@pytest.mark.unit
class TestGetCheckpointPath:
    """Tests for retrieving checkpoint paths."""

    def test_get_checkpoint_path_exists(
        self,
        temp_checkpoint_dir: Path,
        simple_model: nn.Module,
        sample_metadata: CheckpointMetadata,
    ) -> None:
        """Test that get_checkpoint_path returns path for existing checkpoint."""
        manager = CheckpointManager(temp_checkpoint_dir)
        manager.save(simple_model, sample_metadata, name="best")
        path = manager.get_checkpoint_path("best")
        assert path.exists()
        assert path.name == "best.pt"

    def test_get_checkpoint_path_missing_raises(
        self, temp_checkpoint_dir: Path
    ) -> None:
        """Test that get_checkpoint_path raises for missing checkpoint."""
        manager = CheckpointManager(temp_checkpoint_dir)
        with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
            manager.get_checkpoint_path("nonexistent")


# =============================================================================
# Tests: Metadata Preservation
# =============================================================================


@pytest.mark.unit
class TestMetadataPreservation:
    """Tests for metadata preservation through save/load."""

    def test_metadata_fields_preserved(
        self,
        temp_checkpoint_dir: Path,
        simple_model: nn.Module,
        sample_metadata: CheckpointMetadata,
    ) -> None:
        """Test that all metadata fields are preserved."""
        manager = CheckpointManager(temp_checkpoint_dir)
        checkpoint_path = manager.save(simple_model, sample_metadata, name="test")
        loaded_metadata = manager.load(simple_model, checkpoint_path)

        assert loaded_metadata.model_name == sample_metadata.model_name
        assert loaded_metadata.epoch == sample_metadata.epoch
        assert loaded_metadata.val_accuracy == sample_metadata.val_accuracy
        assert loaded_metadata.val_loss == sample_metadata.val_loss
        assert loaded_metadata.config_hash == sample_metadata.config_hash
        assert loaded_metadata.pytorch_version == sample_metadata.pytorch_version

    def test_metadata_json_readable(
        self,
        temp_checkpoint_dir: Path,
        simple_model: nn.Module,
        sample_metadata: CheckpointMetadata,
    ) -> None:
        """Test that metadata JSON is human-readable."""
        manager = CheckpointManager(temp_checkpoint_dir)
        checkpoint_path = manager.save(simple_model, sample_metadata, name="test")
        meta_path = checkpoint_path.parent / checkpoint_path.name.replace(".pt", ".meta.json")

        with meta_path.open("r") as f:
            content = f.read()
            # Should be indented (readable)
            assert "\n" in content
            # Should parse as valid JSON
            parsed = json.loads(content)
            assert "model_name" in parsed


# =============================================================================
# Tests: Edge Cases
# =============================================================================


@pytest.mark.unit
class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_load_strict_mode_true(
        self,
        temp_checkpoint_dir: Path,
        simple_model: nn.Module,
        sample_metadata: CheckpointMetadata,
    ) -> None:
        """Test that load with strict=True requires exact match."""
        manager = CheckpointManager(temp_checkpoint_dir)
        checkpoint_path = manager.save(simple_model, sample_metadata, name="test")

        # Load should succeed with strict=True when model matches
        loaded_metadata = manager.load(simple_model, checkpoint_path, strict=True)
        assert isinstance(loaded_metadata, CheckpointMetadata)

    def test_save_returns_path_type(
        self,
        temp_checkpoint_dir: Path,
        simple_model: nn.Module,
        sample_metadata: CheckpointMetadata,
    ) -> None:
        """Test that save return value is Path object."""
        manager = CheckpointManager(temp_checkpoint_dir)
        result = manager.save(simple_model, sample_metadata, name="test")
        assert isinstance(result, Path)

    def test_checkpoint_manager_path_resolution(
        self, temp_checkpoint_dir: Path
    ) -> None:
        """Test that CheckpointManager resolves paths correctly."""
        manager = CheckpointManager(temp_checkpoint_dir)
        assert manager.checkpoint_dir == temp_checkpoint_dir
        assert isinstance(manager.checkpoint_dir, Path)

    def test_compute_file_checksum_static_method(
        self, temp_checkpoint_dir: Path
    ) -> None:
        """Test that _compute_file_checksum is a static method."""
        # Create a test file
        test_file = temp_checkpoint_dir / "test.txt"
        test_file.write_text("hello world")

        # Call static method directly on class
        checksum = CheckpointManager._compute_file_checksum(test_file)
        assert isinstance(checksum, str)
        assert len(checksum) == 64

        # Verify it matches the actual SHA-256
        expected = hashlib.sha256(b"hello world").hexdigest()
        assert checksum == expected
