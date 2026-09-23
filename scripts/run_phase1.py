#!/usr/bin/env python3
"""Phase 1 experiment orchestrator.

Loads configuration, downloads CIFAR-10, trains model, evaluates on test set,
saves checkpoint, and reports results.

Usage:
    python scripts/run_phase1.py --config experiments/configs/phase1_baseline.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from ml.config import load_config
from ml.datasets.cifar10 import CIFAR10DataModule
from ml.models.cnn import CifarCNN, ModelFactory
from ml.training.checkpoint import CheckpointManager, CheckpointMetadata
from ml.training.evaluator import Evaluator
from ml.training.trainer import Trainer

# =============================================================================
# Setup
# =============================================================================

logger = logging.getLogger(__name__)


def setup_logging(log_dir: Path, log_file: str, level: str = "INFO") -> None:
    """Configure logging to file and console.

    Args:
        log_dir: Directory for log files.
        log_file: Filename for log.
        level: Logging level (DEBUG, INFO, WARNING, ERROR).

    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / log_file

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # File handler
    fh = logging.FileHandler(log_path)
    fh.setLevel(level)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    root_logger.addHandler(fh)
    root_logger.addHandler(ch)

    logger.info(f"Logging to {log_path}")


# =============================================================================
# Experiment
# =============================================================================


def run_experiment(config_path: Path, output_dir: Path | None = None) -> dict[str, Any]:
    """Run Phase 1 experiment end-to-end.

    Args:
        config_path: Path to config YAML file.
        output_dir: Optional output directory for results (defaults to experiments/results).

    Returns:
        Dictionary with experiment results and metadata.

    """
    config_path = Path(config_path)

    # Load configuration
    logger.info(f"Loading config from {config_path}")
    config = load_config(config_path)
    logger.info(f"Experiment name: {config.experiment_name}")
    logger.info(f"Seed: {config.seed}")

    # Setup output directory
    if output_dir is None:
        output_dir = Path("experiments/results/phase1")
    else:
        output_dir = Path(output_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = output_dir / f"{config.experiment_name}_{timestamp}"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Results will be saved to {experiment_dir}")

    # Setup logging
    setup_logging(experiment_dir / "logs", "training.log", config.logging.level)

    # =========================================================================
    # Dataset
    # =========================================================================

    logger.info("Setting up CIFAR-10 dataset...")
    start_data = time.time()

    data_module = CIFAR10DataModule(config.dataset)
    data_module.prepare_data()
    data_module.setup()

    data_duration = time.time() - start_data
    logger.info(f"Dataset setup completed in {data_duration:.2f}s")

    # Verify dataset
    train_loader = data_module.train_loader
    val_loader = data_module.val_loader
    test_loader = data_module.test_loader

    logger.info(f"Train samples: {len(train_loader.dataset)}")  # type: ignore[attr-defined]
    logger.info(f"Val samples: {len(val_loader.dataset)}")  # type: ignore[attr-defined]
    logger.info(f"Test samples: {len(test_loader.dataset)}")  # type: ignore[attr-defined]

    # =========================================================================
    # Model
    # =========================================================================

    logger.info(f"Creating model: {config.model.name}")
    factory = ModelFactory()
    model = factory.create(config.model)
    assert isinstance(model, CifarCNN)
    param_count = model.parameter_count()
    logger.info(f"Model parameters: {param_count:,}")

    # =========================================================================
    # Training
    # =========================================================================

    logger.info("Starting training...")
    start_training = time.time()

    trainer = Trainer(config.training)
    training_result = trainer.train(
        model,
        train_loader,
        val_loader,
        experiment_name=config.experiment_name,
    )

    training_duration = time.time() - start_training
    logger.info(f"Training completed in {training_duration:.2f}s")

    # =========================================================================
    # Test Evaluation
    # =========================================================================

    logger.info("Evaluating on test set...")
    evaluator = Evaluator()
    test_result = evaluator.evaluate(model, test_loader, config.training.device)
    logger.info(f"Test accuracy: {test_result.accuracy:.4f}")
    logger.info(f"Test loss: {test_result.loss:.4f}")

    # =========================================================================
    # Checkpoint
    # =========================================================================

    logger.info("Saving checkpoint...")
    checkpoint_manager = CheckpointManager(config.checkpoint.checkpoint_dir)

    checkpoint_metadata = CheckpointMetadata(
        model_name=config.model.name,
        epoch=training_result.best_epoch,
        val_accuracy=training_result.best_val_accuracy,
        val_loss=0.0,  # Not tracked in result; could be added
        total_epochs_trained=training_result.total_epochs,
        config_hash="",  # Would compute from config file if available
        model_architecture=str(model),
        pytorch_version=torch.__version__,
        checkpoint_version=1,
    )

    checkpoint_path = checkpoint_manager.save(
        model,
        checkpoint_metadata,
        name="best",
    )
    logger.info(f"Checkpoint saved: {checkpoint_path}")

    # Get checksum
    checksum_path = checkpoint_path.parent / checkpoint_path.name.replace(".pt", ".sha256")
    checkpoint_sha256 = checksum_path.read_text().strip()
    logger.info(f"Checkpoint SHA-256: {checkpoint_sha256}")

    # =========================================================================
    # Results
    # =========================================================================

    results = {
        "metadata": {
            "experiment_name": config.experiment_name,
            "seed": config.seed,
            "timestamp": timestamp,
            "config_file": str(config_path),
            "device": config.training.device,
            "pytorch_version": torch.__version__,
        },
        "dataset": {
            "name": config.dataset.name,
            "train_samples": len(train_loader.dataset),  # type: ignore[attr-defined]
            "val_samples": len(val_loader.dataset),  # type: ignore[attr-defined]
            "test_samples": len(test_loader.dataset),  # type: ignore[attr-defined]
            "batch_size": config.dataset.batch_size,
            "val_fraction": config.dataset.val_fraction,
        },
        "model": {
            "name": config.model.name,
            "parameters": param_count,
            "conv_channels": config.model.conv_channels,
            "activation": config.model.activation,
            "dropout": config.model.dropout,
        },
        "training": {
            "total_epochs": training_result.total_epochs,
            "best_epoch": training_result.best_epoch,
            "optimizer": config.training.optimizer,
            "learning_rate": config.training.learning_rate,
            "scheduler": config.training.scheduler,
        },
        "results": {
            "train_loss": training_result.final_train_loss,
            "train_accuracy": training_result.final_train_accuracy,
            "val_accuracy": training_result.best_val_accuracy,
            "test_loss": test_result.loss,
            "test_accuracy": test_result.accuracy,
        },
        "timing": {
            "data_setup_seconds": data_duration,
            "training_seconds": training_duration,
            "total_seconds": time.time() - start_data,
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": checkpoint_sha256,
            "epoch": training_result.best_epoch,
            "val_accuracy": training_result.best_val_accuracy,
        },
    }

    # =========================================================================
    # Report
    # =========================================================================

    logger.info("=" * 80)
    logger.info("PHASE 1 EXPERIMENT RESULTS")
    logger.info("=" * 80)
    logger.info(f"Experiment: {config.experiment_name}")
    logger.info(f"Seed: {config.seed}")
    logger.info(f"Dataset: {config.dataset.name}")
    logger.info(f"Model: {config.model.name} ({param_count:,} parameters)")
    logger.info(f"Training: {config.training.epochs} epochs, {config.training.optimizer}")
    logger.info(f"Completed: {training_result.total_epochs}/{config.training.epochs} epochs")
    logger.info("")
    logger.info(f"Train Loss:     {training_result.final_train_loss:.6f}")
    logger.info(f"Train Accuracy: {training_result.final_train_accuracy:.6f}")
    logger.info(
        f"Val Accuracy:   {training_result.best_val_accuracy:.6f} "
        f"(epoch {training_result.best_epoch})"
    )
    logger.info(f"Test Loss:      {test_result.loss:.6f}")
    logger.info(f"Test Accuracy:  {test_result.accuracy:.6f}")
    logger.info("")
    logger.info(f"Training Duration: {training_duration:.2f}s")
    logger.info(f"Total Duration:    {results['timing']['total_seconds']:.2f}s")
    logger.info("")
    logger.info(f"Checkpoint: {checkpoint_path.name}")
    logger.info(f"SHA-256:    {checkpoint_sha256}")
    logger.info("=" * 80)

    # Save results to JSON
    results_json_path = experiment_dir / "results.json"
    with results_json_path.open("w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_json_path}")

    # Copy checkpoint to results directory
    import shutil
    best_checkpoint_copy = experiment_dir / "best.pt"
    shutil.copy(checkpoint_path, best_checkpoint_copy)
    logger.info(f"Checkpoint copied to {best_checkpoint_copy}")

    # Copy metadata and checksum files
    meta_path = checkpoint_path.parent / checkpoint_path.name.replace(".pt", ".meta.json")
    meta_copy = experiment_dir / "best.meta.json"
    shutil.copy(meta_path, meta_copy)

    checksum_copy = experiment_dir / "best.sha256"
    shutil.copy(checksum_path, checksum_copy)

    return results


# =============================================================================
# CLI
# =============================================================================


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run Phase 1 CIFAR-10 experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for results (default: experiments/results/phase1)",
    )

    args = parser.parse_args()

    try:
        run_experiment(args.config, args.output_dir)
        print("\n✓ Experiment completed successfully!")
        print("Results saved to: experiments/results/phase1")
    except Exception as e:
        logger.exception("Experiment failed")
        print(f"\n✗ Experiment failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
