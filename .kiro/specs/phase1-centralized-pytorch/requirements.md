# Phase 1 — Centralized PyTorch Baseline
## Requirements

**Phase:** 1 of 15  
**Status:** Specification  
**Depends on:** Nothing (first phase)  
**Required before Phase 2:** All acceptance criteria must pass.

---

## Purpose

Phase 1 establishes the foundational machine learning components that every subsequent phase builds on.

The goal is a fully working, testable, configurable **centralized CIFAR-10 image classifier** using PyTorch. No federation, no unlearning, no database, no API.

The code produced here is not throwaway scaffolding. It is the **production ML core** that Flower clients (Phase 2), the contribution tracker (Phase 4), the retraining baseline (Phase 5), and the unlearning pipeline (Phase 6) will all depend on without modification.

---

## Functional Requirements

### FR-1 Dataset

**FR-1.1** The system must download and cache the CIFAR-10 dataset from its canonical source.  
**FR-1.2** The dataset must be split into three non-overlapping subsets: train, validation, and test.  
**FR-1.3** The default split ratio must be configurable (default: 80% train / 10% val / 10% test of the original 50 000 training samples; the 10 000 test samples remain as the held-out test set).  
**FR-1.4** The data directory must be configurable via a configuration file — never hard-coded.  
**FR-1.5** The system must apply appropriate data transforms (normalisation) to all splits.  
**FR-1.6** Optional data augmentation (random crop, horizontal flip) must be configurable and applied to the training split only.  
**FR-1.7** The dataset loader must return PyTorch `DataLoader` objects.  
**FR-1.8** Batch size must be configurable.  
**FR-1.9** The number of DataLoader worker processes must be configurable (default: 2).

### FR-2 Model

**FR-2.1** The system must provide a CNN architecture suitable for CIFAR-10 classification (10 output classes).  
**FR-2.2** The CNN must be defined as a `torch.nn.Module` subclass.  
**FR-2.3** The architecture must be configurable: number of convolutional blocks, filter counts, dropout rate, and activation function must all be settable via configuration.  
**FR-2.4** A `ModelFactory` must create model instances from a configuration dict or object — no direct instantiation in training scripts.  
**FR-2.5** The `ModelFactory` must raise a `ValueError` with a clear message for unknown model names.  
**FR-2.6** The model must be reusable inside Flower `ClientApp` without modification (Phase 2 contract).

### FR-3 Training

**FR-3.1** The system must provide a training loop that trains the model for a configurable number of epochs.  
**FR-3.2** The optimizer must be configurable (default: SGD with momentum; Adam as an option).  
**FR-3.3** Learning rate, momentum, and weight decay must be configurable.  
**FR-3.4** A learning rate scheduler must be optionally configurable (default: StepLR or CosineAnnealingLR).  
**FR-3.5** The training loop must record per-epoch loss and accuracy.  
**FR-3.6** The training loop must detect NaN loss, log a warning, and halt training gracefully.  
**FR-3.7** The training loop must support early stopping based on configurable patience.  
**FR-3.8** The training loop must return a structured `TrainingResult` containing: final training loss, final training accuracy, per-epoch history, total training duration, and device used.  
**FR-3.9** The system must train on CUDA if available and configured; otherwise fall back to CPU.  
**FR-3.10** Device selection must be configurable and logged at startup.

### FR-4 Evaluation

**FR-4.1** The system must provide an evaluator that computes loss and accuracy on a given `DataLoader`.  
**FR-4.2** The evaluator must operate in `torch.no_grad()` mode — it must not modify model weights.  
**FR-4.3** The evaluator must return a structured `EvaluationResult` containing: loss, accuracy, number of samples evaluated, and evaluation duration.  
**FR-4.4** The evaluator must be callable independently (not only inside the training loop).  
**FR-4.5** The evaluator must handle a DataLoader with a single batch without error.

### FR-5 Checkpointing

**FR-5.1** The system must save model checkpoints to a configurable directory.  
**FR-5.2** Each checkpoint must include: model state dict, optimizer state dict, epoch number, training loss, validation accuracy, model version identifier, experiment name, configuration hash, and creation timestamp.  
**FR-5.3** Each saved checkpoint file must have a corresponding SHA-256 checksum stored in a sidecar metadata file.  
**FR-5.4** The checkpoint manager must verify the checksum on load and raise an error if it does not match.  
**FR-5.5** The checkpoint manager must support saving the "best" model (by validation accuracy) without overwriting prior checkpoints.  
**FR-5.6** Loading a checkpoint must return the model state dict and full metadata — callers decide whether to apply the state dict.  
**FR-5.7** The checkpoint manager must never silently overwrite an existing checkpoint at the same path — it must either version or raise.  
**FR-5.8** The checkpoint directory must be excluded from version control.

### FR-6 Configuration

**FR-6.1** All runtime parameters must be expressed in a YAML configuration file.  
**FR-6.2** Configuration must be validated against a Pydantic schema on load; invalid configuration must fail fast with a clear error.  
**FR-6.3** A default configuration file must be provided at `experiments/configs/phase1_baseline.yaml`.  
**FR-6.4** The configuration schema must cover: dataset settings, model settings, training settings, evaluation settings, checkpoint settings, logging settings, and random seed.  
**FR-6.5** The random seed must be set in Python `random`, NumPy, and PyTorch at startup.  
**FR-6.6** The configuration file used must be recorded alongside experiment results.

### FR-7 Logging

**FR-7.1** The system must emit structured log records for all major events using Python's standard `logging` module.  
**FR-7.2** Required log events:

| Event | Level | Fields |
|-------|-------|--------|
| `TRAINING_STARTED` | INFO | experiment, device, epochs, seed |
| `EPOCH_COMPLETED` | INFO | epoch, train_loss, train_acc, val_loss, val_acc, duration_ms |
| `NAN_LOSS_DETECTED` | WARNING | epoch, step |
| `EARLY_STOPPING_TRIGGERED` | INFO | epoch, patience, best_val_acc |
| `CHECKPOINT_SAVED` | INFO | path, checksum, val_acc, epoch |
| `CHECKPOINT_LOADED` | INFO | path, checksum, model_version |
| `TRAINING_COMPLETED` | INFO | total_epochs, best_val_acc, duration_ms |
| `EVALUATION_COMPLETED` | INFO | split, loss, accuracy, n_samples, duration_ms |

**FR-7.3** Logs must never include raw tensor values, gradient values, or pixel data.  
**FR-7.4** Log output must go to both console (INFO+) and a rotating file handler (DEBUG+).

---

## Non-Functional Requirements

### NFR-1 Correctness
- Training must converge on CIFAR-10. A correctly configured run of 20 epochs must achieve validation accuracy consistently above 60% (a simple CNN baseline, not a state-of-the-art result). This threshold is informational — the system does not hard-code it as a pass/fail criterion.

### NFR-2 Reproducibility
- Running the same configuration file with the same random seed on the same hardware must produce identical results across runs.

### NFR-3 Performance
- A 5-epoch training run on CIFAR-10 with CPU must complete within a reasonable time (< 30 minutes on a modern machine without GPU). The system must not hang indefinitely.

### NFR-4 Modularity
- The `ml/` package must have no import dependencies on `apps/`, `contribution/`, `unlearning/`, `evaluation/`, or `scheduler/`. It must stand alone.

### NFR-5 Testability
- Every public function and class must be unit-testable in isolation with mocked or synthetic data.
- Tests must not require downloading CIFAR-10 — a synthetic dataset fixture must be available.

### NFR-6 Portability
- The code must run on Windows, Linux, and macOS.
- Path handling must use `pathlib.Path`.

---

## Out of Scope for Phase 1

The following are explicitly **not** part of Phase 1:

- Federated learning (Flower) — Phase 2
- Data partitioning across multiple clients — Phase 3
- Contribution tracking — Phase 4
- Unlearning — Phase 6
- FastAPI — Phase 11
- PostgreSQL — Phase 12
- React dashboard — Phase 13
- Docker — Phase 14

If any of these appear in Phase 1 code, they are a scope violation.

---

## Acceptance Criteria

Phase 1 is complete when **all** of the following are true:

| # | Criterion | How to verify |
|---|-----------|--------------|
| AC-1 | CIFAR-10 downloads and caches on first run | Run `python -m ml.datasets.cifar10` |
| AC-2 | Second run uses cache, does not re-download | Run again, observe no network activity |
| AC-3 | Model initialises and produces output shape `(B, 10)` | `test_cnn.py` |
| AC-4 | ModelFactory creates model from config dict | `test_cnn.py` |
| AC-5 | Training loop runs for 5 epochs without error | `test_trainer.py` or `scripts/run_phase1.py` |
| AC-6 | Per-epoch loss and accuracy are logged | Inspect log output |
| AC-7 | Validation is performed after each epoch | Inspect log output |
| AC-8 | NaN loss detection halts training and logs WARNING | `test_trainer.py` edge case |
| AC-9 | Evaluator returns correct accuracy on known data | `test_evaluator.py` |
| AC-10 | Checkpoint is saved after training | File exists on disk |
| AC-11 | Checkpoint includes SHA-256 checksum | Inspect sidecar file |
| AC-12 | Checkpoint loads and model produces same output | `test_checkpoint.py` |
| AC-13 | Checksum mismatch raises error on load | `test_checkpoint.py` edge case |
| AC-14 | Configuration loads and validates from YAML | `test_config.py` |
| AC-15 | Invalid config raises `ValidationError` with clear message | `test_config.py` edge case |
| AC-16 | Random seed produces reproducible results | Run twice, compare checkpoints |
| AC-17 | All Phase 1 unit tests pass | `pytest tests/ -m unit` |
| AC-18 | All Phase 1 integration tests pass | `pytest tests/ -m integration` |
| AC-19 | `ml/` has no imports from other project packages | `ruff check` or import graph |
| AC-20 | A 5-epoch run produces a valid checkpoint and logs | End-to-end script |
