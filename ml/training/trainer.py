"""Training loop for centralized Phase 1 CIFAR-10 baseline.

Provides Trainer for end-to-end training with:
- Per-epoch loss and accuracy tracking
- Validation each epoch
- Early stopping support
- Device handling (CPU/CUDA)
- Configurable optimizer and learning rate scheduler
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader

from ml.config import TrainingConfig
from ml.training.evaluator import EvaluationResult, Evaluator

if TYPE_CHECKING:
    from torch.optim import Optimizer

logger = logging.getLogger(__name__)

# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class EpochRecord:
    """Training metrics for a single epoch.

    Attributes:
        epoch: Epoch number (1-indexed).
        train_loss: Average training loss.
        train_accuracy: Training accuracy (fraction correct).
        val_loss: Validation loss.
        val_accuracy: Validation accuracy (fraction correct).
        duration_seconds: Time taken for the epoch.

    """

    epoch: int
    train_loss: float
    train_accuracy: float
    val_loss: float
    val_accuracy: float
    duration_seconds: float


@dataclass
class TrainingResult:
    """Result from a complete training run.

    Attributes:
        experiment_name: Name of the experiment.
        model_name: Name of the model used.
        total_epochs: Total epochs completed.
        best_epoch: Epoch with best validation accuracy.
        best_val_accuracy: Best validation accuracy achieved.
        final_train_loss: Training loss at final epoch.
        final_train_accuracy: Training accuracy at final epoch.
        history: List of EpochRecord for each completed epoch.
        total_duration_seconds: Total time for training.
        device: Device used (cpu or cuda).
        checkpoint_path: Path to best checkpoint (if saved).

    """

    experiment_name: str
    model_name: str
    total_epochs: int
    best_epoch: int
    best_val_accuracy: float
    final_train_loss: float
    final_train_accuracy: float
    history: list[EpochRecord]
    total_duration_seconds: float
    device: str
    checkpoint_path: Path | None = None


# =============================================================================
# Helper Functions
# =============================================================================


def _resolve_device(device_config: str) -> torch.device:
    """Resolve device from configuration string.

    Args:
        device_config: "auto", "cpu", or "cuda".

    Returns:
        torch.device object.

    Raises:
        ValueError: If device_config is "cuda" but CUDA not available.

    """
    if device_config == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device_config == "cpu":
        device = torch.device("cpu")
    elif device_config == "cuda":
        if not torch.cuda.is_available():
            msg = "CUDA requested but not available"
            raise ValueError(msg)
        device = torch.device("cuda")
    else:
        msg = f'Invalid device config: {device_config}. Must be "auto", "cpu", or "cuda".'
        raise ValueError(msg)

    logger.info(f"Using device: {device}")
    return device


def _build_optimizer(model: nn.Module, config: TrainingConfig) -> Optimizer:
    """Build optimizer from configuration.

    Args:
        model: PyTorch model to optimize.
        config: TrainingConfig with optimizer, learning_rate, momentum, weight_decay.

    Returns:
        Optimizer instance.

    Raises:
        ValueError: If optimizer name is unknown.

    """
    if config.optimizer == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
    elif config.optimizer == "adam":
        return optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    else:
        msg = f'Unknown optimizer: {config.optimizer}. Must be "sgd" or "adam".'
        raise ValueError(msg)


def _build_scheduler(
    optimizer: Optimizer,
    config: TrainingConfig,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    """Build learning rate scheduler from configuration.

    Args:
        optimizer: PyTorch optimizer.
        config: TrainingConfig with scheduler, epochs.

    Returns:
        LRScheduler or None if no scheduler is configured.

    Raises:
        ValueError: If scheduler name is unknown.

    """
    if config.scheduler == "cosine":
        return CosineAnnealingLR(optimizer, T_max=config.epochs)
    elif config.scheduler == "step":
        return StepLR(optimizer, step_size=5, gamma=0.5)
    elif config.scheduler is None:
        return None
    else:
        msg = f'Unknown scheduler: {config.scheduler}. Must be "cosine", "step", or None.'
        raise ValueError(msg)


# =============================================================================
# Trainer
# =============================================================================


class Trainer:
    """Trainer for centralized PyTorch model training.

    Orchestrates the training loop:
    - Configures optimizer and scheduler
    - Trains for configured epochs
    - Evaluates on validation set each epoch
    - Implements early stopping
    - Tracks metrics in EpochRecord
    """

    def __init__(self, config: TrainingConfig) -> None:
        """Initialise Trainer.

        Args:
            config: TrainingConfig with learning_rate, epochs, scheduler, etc.

        """
        self.config = config
        self.evaluator = Evaluator(criterion=nn.CrossEntropyLoss())

    def train(
        self,
        model: nn.Module,
        train_loader: DataLoader,  # type: ignore[type-arg]
        val_loader: DataLoader,  # type: ignore[type-arg]
        experiment_name: str = "experiment",
    ) -> TrainingResult:
        """Run the full training loop.

        Args:
            model: PyTorch model to train.
            train_loader: DataLoader with training batches.
            val_loader: DataLoader with validation batches.
            experiment_name: Name of the experiment for logging.

        Returns:
            TrainingResult with full training history and metrics.

        """
        start_time = time.time()

        # Resolve device
        device = _resolve_device(self.config.device)

        # Move model to device
        model.to(device)

        # Build optimizer and scheduler
        optimizer = _build_optimizer(model, self.config)
        scheduler = _build_scheduler(optimizer, self.config)

        # Training state
        history: list[EpochRecord] = []
        best_val_accuracy = 0.0
        best_epoch = 0
        patience_counter = 0

        # Main training loop
        for epoch in range(1, self.config.epochs + 1):
            epoch_start = time.time()

            # Training phase
            train_loss, train_accuracy = self._train_one_epoch(
                model, train_loader, optimizer, device
            )

            # Check for NaN loss
            if not torch.isfinite(torch.tensor(train_loss)):
                logger.warning("NAN_LOSS_DETECTED")
                break

            # Validation phase
            val_result: EvaluationResult = self.evaluator.evaluate(
                model, val_loader, device
            )

            # Step scheduler
            if scheduler is not None:
                scheduler.step()

            # Record epoch metrics
            epoch_duration = time.time() - epoch_start
            epoch_record = EpochRecord(
                epoch=epoch,
                train_loss=train_loss,
                train_accuracy=train_accuracy,
                val_loss=val_result.loss,
                val_accuracy=val_result.accuracy,
                duration_seconds=epoch_duration,
            )
            history.append(epoch_record)

            logger.info("EPOCH_COMPLETED")

            # Early stopping logic
            if val_result.accuracy > best_val_accuracy:
                best_val_accuracy = val_result.accuracy
                best_epoch = epoch
                patience_counter = 0
                logger.info(f"Validation accuracy improved: {best_val_accuracy:.4f}")
            else:
                patience_counter += 1
                if (
                    self.config.early_stopping_patience
                    and patience_counter >= self.config.early_stopping_patience
                ):
                    logger.info("EARLY_STOPPING_TRIGGERED")
                    break

        # Total training time
        total_duration = time.time() - start_time

        # Get final metrics from last epoch
        final_train_loss = history[-1].train_loss if history else 0.0
        final_train_accuracy = history[-1].train_accuracy if history else 0.0

        # Create result
        model_name_attr = (
            self.config.model_name
            if hasattr(self.config, "model_name")
            else "cifar_cnn"
        )
        result = TrainingResult(
            experiment_name=experiment_name,
            model_name=model_name_attr,
            total_epochs=len(history),
            best_epoch=best_epoch,
            best_val_accuracy=best_val_accuracy,
            final_train_loss=final_train_loss,
            final_train_accuracy=final_train_accuracy,
            history=history,
            total_duration_seconds=total_duration,
            device=str(device),
            checkpoint_path=None,
        )

        return result

    def _train_one_epoch(
        self,
        model: nn.Module,
        train_loader: DataLoader,  # type: ignore[type-arg]
        optimizer: Optimizer,
        device: torch.device,
    ) -> tuple[float, float]:
        """Train for one epoch.

        Args:
            model: PyTorch model.
            train_loader: DataLoader with training batches.
            optimizer: Optimizer for parameter updates.
            device: Device to train on.

        Returns:
            Tuple of (average_loss, accuracy).

        """
        model.train()
        criterion = nn.CrossEntropyLoss()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch_images, batch_labels in train_loader:
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)

            # Forward pass
            logits = model(batch_images)
            loss = criterion(logits, batch_labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate metrics
            total_loss += loss.item() * batch_labels.size(0)
            predictions = logits.argmax(dim=1)
            total_correct += (predictions == batch_labels).sum().item()
            total_samples += batch_labels.size(0)

        # Compute averages
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0

        return avg_loss, accuracy
