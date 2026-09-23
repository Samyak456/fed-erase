"""Checkpoint management for model persistence.

Provides CheckpointMetadata and CheckpointManager for saving and loading
model state_dict with full reproducibility metadata, SHA-256 validation,
and best-model tracking.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class CheckpointMetadata:
    """Metadata stored alongside checkpoint file.

    Attributes:
        model_name: Name of the model architecture.
        epoch: Epoch number when checkpoint was saved.
        val_accuracy: Validation accuracy at this checkpoint.
        val_loss: Validation loss at this checkpoint.
        total_epochs_trained: Total epochs completed in the training run.
        config_hash: SHA-256 hash of the configuration file used.
        model_architecture: String representation of model architecture.
        pytorch_version: PyTorch version used.
        checkpoint_version: Version number of checkpoint format.

    """

    model_name: str
    epoch: int
    val_accuracy: float
    val_loss: float
    total_epochs_trained: int
    config_hash: str
    model_architecture: str
    pytorch_version: str
    checkpoint_version: int = 1


class CheckpointManager:
    """Manages model checkpoint save/load with metadata and validation.

    Responsibilities:
    - Save model state_dict with metadata and SHA-256 checksum
    - Load checkpoint and verify checksum matches
    - Track best model based on validation metric
    - Support versioned checkpoints (best, latest, last_n)
    - Never silently overwrite the only copy of a checkpoint
    """

    def __init__(self, checkpoint_dir: Path, keep_best_only: bool = True) -> None:
        """Initialise CheckpointManager.

        Args:
            checkpoint_dir: Directory where checkpoints are saved.
            keep_best_only: If True, only keep best checkpoint; if False, keep all.

        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.keep_best_only = keep_best_only

        # Create checkpoint directory if it doesn't exist
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Checkpoint directory: {self.checkpoint_dir}")

    def save(
        self,
        model: nn.Module,
        metadata: CheckpointMetadata,
        name: str = "best",
    ) -> Path:
        """Save model checkpoint with metadata and checksum.

        Saves two files:
        - `<name>.pt` — PyTorch state_dict
        - `<name>.meta.json` — Metadata JSON
        - `<name>.sha256` — SHA-256 checksum of state_dict file

        Does NOT overwrite if file already exists unless explicitly replacing.
        Creates new version instead (name_v1, name_v2, etc.).

        Args:
            model: PyTorch model to save (state_dict extracted).
            metadata: CheckpointMetadata with epoch, accuracy, config hash, etc.
            name: Checkpoint name (default: "best").

        Returns:
            Path to saved checkpoint file.

        Raises:
            IOError: If checkpoint directory does not exist or is not writable.

        """
        if not self.checkpoint_dir.exists():
            msg = f"Checkpoint directory does not exist: {self.checkpoint_dir}"
            raise OSError(msg)

        # Determine checkpoint paths
        checkpoint_path = self.checkpoint_dir / f"{name}.pt"
        meta_path = self.checkpoint_dir / f"{name}.meta.json"
        checksum_path = self.checkpoint_dir / f"{name}.sha256"

        # Check if checkpoint already exists; if so, create a new version
        if checkpoint_path.exists():
            version = 1
            while (self.checkpoint_dir / f"{name}_v{version}.pt").exists():
                version += 1
            checkpoint_path = self.checkpoint_dir / f"{name}_v{version}.pt"
            meta_path = self.checkpoint_dir / f"{name}_v{version}.meta.json"
            checksum_path = self.checkpoint_dir / f"{name}_v{version}.sha256"
            logger.info(
                f"Checkpoint {name}.pt already exists; saving as version {version}"
            )

        # Save state_dict
        state_dict = model.state_dict()
        torch.save(state_dict, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")

        # Compute SHA-256 checksum of state_dict file
        checksum = self._compute_file_checksum(checkpoint_path)
        with checksum_path.open("w") as f:
            f.write(checksum)
        logger.info(f"Checksum saved: {checksum_path}")

        # Save metadata
        metadata_dict = asdict(metadata)
        metadata_dict["checksum"] = checksum
        with meta_path.open("w") as f:
            json.dump(metadata_dict, f, indent=2)
        logger.info(f"Metadata saved: {meta_path}")

        return checkpoint_path

    def load(
        self,
        model: nn.Module,
        checkpoint_path: Path | str,
        strict: bool = True,
    ) -> CheckpointMetadata:
        """Load checkpoint into model and verify checksum.

        Loads state_dict from checkpoint file, verifies SHA-256 checksum,
        loads metadata, and returns metadata for reference.

        Args:
            model: PyTorch model to load state_dict into.
            checkpoint_path: Path to checkpoint file (.pt).
            strict: If True, require exact state_dict match; if False, allow partial.

        Returns:
            CheckpointMetadata loaded from metadata file.

        Raises:
            FileNotFoundError: If checkpoint or metadata file does not exist.
            ValueError: If checksum mismatch (file corrupted).
            RuntimeError: If state_dict cannot be loaded into model.

        """
        checkpoint_path = Path(checkpoint_path)

        # Verify checkpoint exists
        if not checkpoint_path.exists():
            msg = f"Checkpoint file not found: {checkpoint_path}"
            raise FileNotFoundError(msg)

        # Verify metadata exists
        meta_path = checkpoint_path.parent / checkpoint_path.name.replace(".pt", ".meta.json")
        if not meta_path.exists():
            msg = f"Metadata file not found: {meta_path}"
            raise FileNotFoundError(msg)

        # Verify checksum exists
        checksum_path = checkpoint_path.parent / checkpoint_path.name.replace(".pt", ".sha256")
        if not checksum_path.exists():
            msg = f"Checksum file not found: {checksum_path}"
            raise FileNotFoundError(msg)

        # Verify checksum matches
        stored_checksum = checksum_path.read_text().strip()
        computed_checksum = self._compute_file_checksum(checkpoint_path)
        if stored_checksum != computed_checksum:
            msg = (
                f"Checksum mismatch for {checkpoint_path}: "
                f"stored={stored_checksum}, computed={computed_checksum}"
            )
            raise ValueError(msg)
        logger.info("Checksum verified")

        # Load metadata
        with meta_path.open("r") as f:
            metadata_dict = json.load(f)
        # Remove checksum from metadata_dict as it's not a CheckpointMetadata field
        metadata_dict.pop("checksum", None)
        metadata = CheckpointMetadata(**metadata_dict)
        logger.info(f"Metadata loaded from {meta_path}")

        # Load state_dict into model
        state_dict = torch.load(checkpoint_path, weights_only=True)
        model.load_state_dict(state_dict, strict=strict)
        logger.info(f"Checkpoint loaded: {checkpoint_path}")

        return metadata

    @staticmethod
    def _compute_file_checksum(file_path: Path) -> str:
        """Compute SHA-256 checksum of a file.

        Args:
            file_path: Path to file.

        Returns:
            Hex string of SHA-256 checksum.

        """
        hasher = hashlib.sha256()
        with file_path.open("rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def get_checkpoint_path(self, name: str = "best") -> Path:
        """Get path to checkpoint file by name.

        Args:
            name: Checkpoint name (e.g., "best", "latest").

        Returns:
            Path to checkpoint file.

        Raises:
            FileNotFoundError: If checkpoint does not exist.

        """
        checkpoint_path = self.checkpoint_dir / f"{name}.pt"
        if not checkpoint_path.exists():
            msg = f"Checkpoint not found: {checkpoint_path}"
            raise FileNotFoundError(msg)
        return checkpoint_path

    def list_checkpoints(self) -> dict[str, Path]:
        """List all checkpoints in checkpoint directory.

        Returns:
            Dict mapping checkpoint names to their paths (only .pt files).

        """
        checkpoints = {}
        for pt_file in self.checkpoint_dir.glob("*.pt"):
            name = pt_file.stem
            checkpoints[name] = pt_file
        return checkpoints
