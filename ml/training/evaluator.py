"""Model evaluation during training and validation.

Provides EvaluationResult and Evaluator for computing validation metrics
(loss, accuracy) on a given DataLoader without computing gradients.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Result from model evaluation.

    Attributes:
        loss: Average loss across all samples in the loader.
        accuracy: Fraction of correct predictions (in [0.0, 1.0]).
        n_samples: Total number of samples evaluated.
        duration_seconds: Time taken for evaluation.

    """

    loss: float
    accuracy: float
    n_samples: int
    duration_seconds: float


class Evaluator:
    """Evaluates a model on a DataLoader without computing gradients.

    The evaluator:
    - Sets the model to eval() mode
    - Runs inference under torch.no_grad()
    - Computes loss and accuracy
    - Does NOT modify model weights
    - Optionally restores model to train() mode after evaluation
    """

    def __init__(self, criterion: nn.Module | None = None) -> None:
        """Initialise the Evaluator.

        Args:
            criterion: Loss function. Defaults to nn.CrossEntropyLoss if None.

        """
        self.criterion = criterion or nn.CrossEntropyLoss()

    def evaluate(
        self,
        model: nn.Module,
        loader: DataLoader,  # type: ignore[type-arg]
        device: torch.device | str = "cpu",
    ) -> EvaluationResult:
        """Evaluate model on a DataLoader.

        Sets model to eval() mode, runs inference under torch.no_grad(),
        and computes loss and accuracy. Does not modify model weights.

        Args:
            model: PyTorch model to evaluate.
            loader: DataLoader with (images, labels) batches.
            device: Device to evaluate on ("cpu", "cuda", or torch.device).

        Returns:
            EvaluationResult with loss, accuracy, n_samples, duration_seconds.

        """
        if isinstance(device, str):
            device = torch.device(device)

        # Store original mode to restore later
        was_training = model.training

        model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        start_time = time.time()

        try:
            with torch.no_grad():
                for batch_images, batch_labels in loader:
                    batch_images = batch_images.to(device)
                    batch_labels = batch_labels.to(device)

                    # Forward pass
                    logits = model(batch_images)
                    loss = self.criterion(logits, batch_labels)

                    # Accumulate loss
                    total_loss += loss.item() * batch_labels.size(0)

                    # Accumulate accuracy
                    predictions = logits.argmax(dim=1)
                    total_correct += (predictions == batch_labels).sum().item()
                    total_samples += batch_labels.size(0)
        finally:
            # Always restore original mode
            if was_training:
                model.train()

        duration_seconds = time.time() - start_time

        # Compute average metrics
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0

        logger.info("EVALUATION_COMPLETED")

        return EvaluationResult(
            loss=avg_loss,
            accuracy=accuracy,
            n_samples=total_samples,
            duration_seconds=duration_seconds,
        )
