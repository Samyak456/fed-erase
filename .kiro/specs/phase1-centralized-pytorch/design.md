# Phase 1 — Centralized PyTorch Baseline
## Architecture and Design

**Phase:** 1 of 15  
**Status:** Specification  

---

## Overview

Phase 1 produces a self-contained ML package at `ml/` plus supporting configuration and
scripts. The design prioritises reusability (Phase 2 will drop the same model into Flower
clients without changes) and testability (every component is independently mockable).

---

## Directory Structure

```
fed-erase/
│
├── pyproject.toml                    # project metadata and dependencies
├── .env.example                      # env var documentation
├── Makefile                          # convenience targets
│
├── config/
│   └── defaults.yaml                 # project-wide defaults
│
├── experiments/
│   └── configs/
│       └── phase1_baseline.yaml      # Phase 1 experiment config
│
├── ml/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── cnn.py                    # CifarCNN architecture
│   │   └── factory.py                # ModelFactory
│   ├── datasets/
│   │   ├── __init__.py
│   │   ├── cifar10.py                # CIFAR-10 loader and split logic
│   │   ├── partitioning.py           # IID / Dirichlet (stubbed in Phase 1)
│   │   └── transforms.py             # Normalisation and augmentation
│   └── training/
│       ├── __init__.py
│       ├── trainer.py                # Trainer class
│       ├── evaluator.py              # Evaluator class
│       └── checkpoint.py             # CheckpointManager
│
├── scripts/
│   └── run_phase1.py                 # End-to-end Phase 1 entry point
│
├── tests/
│   ├── conftest.py                   # shared fixtures
│   ├── ml/
│   │   ├── __init__.py
│   │   ├── datasets/
│   │   │   ├── test_cifar10.py
│   │   │   └── test_partitioning.py
│   │   ├── models/
│   │   │   └── test_cnn.py
│   │   └── training/
│   │       ├── test_trainer.py
│   │       ├── test_evaluator.py
│   │       └── test_checkpoint.py
│
└── docs/
    └── algorithm.md                  # algorithmic assumptions (started in Phase 1)
```

---

## Component Designs

### 1. Configuration (`config/`, `experiments/configs/`)

**Schema class:** `FedEraseConfig` (Pydantic `BaseSettings`)

Sub-schemas:

```
FedEraseConfig
├── DatasetConfig
│   ├── name: str = "cifar10"
│   ├── data_dir: Path
│   ├── val_fraction: float = 0.1
│   ├── batch_size: int = 64
│   ├── num_workers: int = 2
│   └── augment_train: bool = True
│
├── ModelConfig
│   ├── name: str = "cifar_cnn"
│   ├── num_classes: int = 10
│   ├── conv_channels: list[int] = [32, 64, 128]
│   ├── dropout: float = 0.3
│   └── activation: str = "relu"
│
├── TrainingConfig
│   ├── epochs: int = 20
│   ├── optimizer: str = "sgd"
│   ├── learning_rate: float = 0.01
│   ├── momentum: float = 0.9
│   ├── weight_decay: float = 1e-4
│   ├── scheduler: str | None = "cosine"
│   ├── early_stopping_patience: int | None = 5
│   └── device: str = "auto"   # "auto" | "cpu" | "cuda"
│
├── CheckpointConfig
│   ├── checkpoint_dir: Path
│   ├── save_best_only: bool = True
│   └── keep_last_n: int = 3
│
├── LoggingConfig
│   ├── level: str = "INFO"
│   ├── log_dir: Path
│   └── log_file: str = "phase1.log"
│
└── seed: int = 42
    experiment_name: str = "phase1_baseline"
```

Configuration is loaded via a `load_config(path: Path) -> FedEraseConfig` function that:
1. Reads the YAML file.
2. Validates with Pydantic.
3. Resolves relative paths against the project root.
4. Sets all random seeds immediately.

---

### 2. Dataset Loader (`ml/datasets/cifar10.py`)

**Class:** `CIFAR10DataModule`

```python
class CIFAR10DataModule:
    def __init__(self, config: DatasetConfig) -> None: ...

    def prepare_data(self) -> None:
        """Download CIFAR-10 if not cached. Called once."""

    def setup(self) -> None:
        """Build train/val/test splits and DataLoaders."""

    @property
    def train_loader(self) -> DataLoader: ...

    @property
    def val_loader(self) -> DataLoader: ...

    @property
    def test_loader(self) -> DataLoader: ...

    @property
    def num_classes(self) -> int: ...

    @property
    def class_names(self) -> list[str]: ...
```

**Split logic:**

```
Original CIFAR-10 training set (50 000 samples)
    │
    ├── Train subset:  floor(50000 * (1 - val_fraction))  indices
    └── Val subset:    remaining indices

Original CIFAR-10 test set (10 000 samples)
    └── Test set:      all 10 000 (unchanged)
```

Splits are created using `torch.utils.data.Subset` with deterministic index selection
(shuffled once using the configured random seed, then fixed).

**Transforms** (`ml/datasets/transforms.py`):

```python
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

def get_train_transforms(augment: bool) -> transforms.Compose: ...
def get_eval_transforms() -> transforms.Compose: ...
```

Train transforms (when `augment=True`):
1. `RandomCrop(32, padding=4)`
2. `RandomHorizontalFlip()`
3. `ToTensor()`
4. `Normalize(CIFAR10_MEAN, CIFAR10_STD)`

Eval transforms:
1. `ToTensor()`
2. `Normalize(CIFAR10_MEAN, CIFAR10_STD)`

---

### 3. CNN Architecture (`ml/models/cnn.py`)

**Class:** `CifarCNN(nn.Module)`

Architecture pattern: `[Conv → BN → ReLU → Pool] × N → Flatten → FC → Dropout → FC`

Default architecture (3 conv blocks):

```
Input:  (B, 3, 32, 32)
│
├── Block 1: Conv2d(3→32, 3×3, pad=1) → BN → ReLU → MaxPool(2×2)
│   Output: (B, 32, 16, 16)
│
├── Block 2: Conv2d(32→64, 3×3, pad=1) → BN → ReLU → MaxPool(2×2)
│   Output: (B, 64, 8, 8)
│
├── Block 3: Conv2d(64→128, 3×3, pad=1) → BN → ReLU → MaxPool(2×2)
│   Output: (B, 128, 4, 4)
│
├── Flatten → (B, 2048)
│
├── FC(2048 → 256) → ReLU → Dropout(p)
│
└── FC(256 → 10)
    Output: (B, 10)  [logits]
```

The number of conv blocks and filter sizes are driven by `ModelConfig.conv_channels`.
The final spatial size and FC input dimension are computed dynamically, not hard-coded.

**Design decision:** The model outputs raw logits (not softmax). Loss is `nn.CrossEntropyLoss`
which expects logits. Evaluation computes `softmax` before `argmax` for accuracy.

---

### 4. Model Factory (`ml/models/factory.py`)

**Class:** `ModelFactory`

```python
_REGISTRY: dict[str, type[nn.Module]] = {
    "cifar_cnn": CifarCNN,
}

class ModelFactory:
    @staticmethod
    def create(config: ModelConfig) -> nn.Module:
        """Create a model instance from configuration.

        Raises:
            ValueError: If config.name is not in the registry.
        """

    @staticmethod
    def register(name: str, cls: type[nn.Module]) -> None:
        """Register a new model class. Used in tests and extensions."""
```

This registry pattern allows Phase 2+ to add ResNet or other models without modifying
the factory's core logic.

---

### 5. Trainer (`ml/training/trainer.py`)

**Dataclass:** `TrainingResult`

```python
@dataclass
class TrainingResult:
    experiment_name: str
    model_name: str
    total_epochs: int
    best_epoch: int
    best_val_accuracy: float
    final_train_loss: float
    final_train_accuracy: float
    history: list[EpochRecord]   # per-epoch metrics
    total_duration_seconds: float
    device: str
    checkpoint_path: Path | None
```

**Dataclass:** `EpochRecord`

```python
@dataclass
class EpochRecord:
    epoch: int
    train_loss: float
    train_accuracy: float
    val_loss: float
    val_accuracy: float
    duration_seconds: float
```

**Class:** `Trainer`

```python
class Trainer:
    def __init__(self, config: TrainingConfig, checkpoint_manager: CheckpointManager) -> None: ...

    def train(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        experiment_name: str,
    ) -> TrainingResult:
        """Run the full training loop.

        Does NOT modify the model in-place after training — returns a result.
        The caller is responsible for saving the model if desired.
        """
```

**Training loop pseudocode:**

```
set_seeds(config.seed)
device = resolve_device(config.device)
model.to(device)
optimizer = build_optimizer(model, config)
scheduler = build_scheduler(optimizer, config)

best_val_acc = 0.0
patience_counter = 0
history = []

for epoch in 1..config.epochs:
    train_loss, train_acc = _train_one_epoch(model, train_loader, optimizer, device)

    if isnan(train_loss):
        log WARNING NAN_LOSS_DETECTED
        break

    val_loss, val_acc = evaluator.evaluate(model, val_loader)
    scheduler.step()

    history.append(EpochRecord(...))
    log INFO EPOCH_COMPLETED

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        checkpoint_manager.save_best(model, optimizer, metadata)
        patience_counter = 0
    else:
        patience_counter += 1

    if config.early_stopping_patience and patience_counter >= patience:
        log INFO EARLY_STOPPING_TRIGGERED
        break

return TrainingResult(...)
```

---

### 6. Evaluator (`ml/training/evaluator.py`)

**Dataclass:** `EvaluationResult`

```python
@dataclass
class EvaluationResult:
    loss: float
    accuracy: float
    n_samples: int
    duration_seconds: float
```

**Class:** `Evaluator`

```python
class Evaluator:
    def __init__(self, criterion: nn.Module | None = None) -> None: ...

    def evaluate(self, model: nn.Module, loader: DataLoader, device: str = "cpu") -> EvaluationResult:
        """Evaluate model on the given DataLoader.

        Runs in torch.no_grad() context. Does not modify model weights.
        """
```

---

### 7. Checkpoint Manager (`ml/training/checkpoint.py`)

**Dataclass:** `CheckpointMetadata`

```python
@dataclass
class CheckpointMetadata:
    model_version: str          # e.g. "phase1_baseline_v1"
    experiment_name: str
    epoch: int
    val_accuracy: float
    train_loss: float
    config_hash: str            # SHA-256 of the config YAML
    created_at: str             # ISO 8601
    checksum: str               # SHA-256 of the .pt file
    path: Path
```

**Class:** `CheckpointManager`

```python
class CheckpointManager:
    def __init__(self, config: CheckpointConfig, experiment_name: str) -> None: ...

    def save(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        metadata: CheckpointMetadata,
    ) -> Path:
        """Save checkpoint and sidecar metadata JSON. Never overwrites an existing path."""

    def save_best(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        metadata: CheckpointMetadata,
    ) -> Path:
        """Save as 'best' checkpoint. Keeps previous best under a versioned name."""

    def load(self, path: Path) -> tuple[dict, dict, CheckpointMetadata]:
        """Load and verify checkpoint. Returns (state_dict, optimizer_state, metadata).

        Raises:
            FileNotFoundError: If path does not exist.
            ChecksumMismatchError: If SHA-256 of file does not match stored checksum.
        """

    def list_checkpoints(self) -> list[CheckpointMetadata]:
        """List all checkpoints in the configured directory, sorted by creation time."""

    def cleanup(self, keep_last_n: int) -> None:
        """Delete old checkpoints, keeping the N most recent plus the best."""
```

**Checksum protocol:**
1. Write `.pt` file.
2. Compute SHA-256 of `.pt` file bytes.
3. Write `<filename>.meta.json` with all metadata including the checksum.
4. On load: re-compute SHA-256 of `.pt` file, compare with stored value.

---

### 8. Entry Point Script (`scripts/run_phase1.py`)

```python
"""Phase 1 end-to-end training script.

Usage:
    python scripts/run_phase1.py --config experiments/configs/phase1_baseline.yaml
"""
```

Flow:
1. Parse `--config` argument.
2. `load_config(path)` → validate, set seeds, configure logging.
3. `CIFAR10DataModule(config.dataset)` → `prepare_data()` → `setup()`.
4. `ModelFactory.create(config.model)`.
5. `CheckpointManager(config.checkpoint, config.experiment_name)`.
6. `Trainer(config.training, checkpoint_manager)`.
7. `trainer.train(model, train_loader, val_loader, experiment_name)`.
8. `evaluator.evaluate(model, test_loader)` → log final test accuracy.
9. Print summary table to stdout.
10. Exit 0 on success, 1 on error.

---

## Data Flow Diagram

```
phase1_baseline.yaml
        │
        ▼
  load_config()
        │
   FedEraseConfig
        │
   ┌────┴────┐
   │         │
   ▼         ▼
CIFAR10   ModelFactory
DataModule  .create()
   │           │
   │        CifarCNN
   │           │
   ▼           ▼
DataLoaders  model (nn.Module)
   │           │
   └─────┬─────┘
         │
         ▼
      Trainer
         │
    ┌────┴────┐
    │         │
    ▼         ▼
 train()   Evaluator
    │      .evaluate()  ← per epoch (val)
    │           │
    └────┬──────┘
         │
         ▼
  CheckpointManager
    .save_best()
         │
         ▼
     .pt + .meta.json
```

---

## Key Design Decisions

### Decision 1 — Logits vs. Probabilities

The model outputs raw logits. `nn.CrossEntropyLoss` is applied in the training loop.
Accuracy is computed from `argmax(logits)` after `softmax`.

**Rationale:** Standard PyTorch practice. Avoids numerical instability from double-softmax.

### Decision 2 — Subset splits via indices, not random sampling

Train/val split uses deterministic index assignment (shuffled once at `setup()` time using
the configured seed), not random sampling on every epoch.

**Rationale:** Guarantees reproducibility. Ensures the same images are always in the same split
when the same seed is used.

### Decision 3 — Checkpoint sidecar metadata file

Metadata is stored in a separate JSON file alongside the `.pt` file, not embedded in the
`torch.save()` payload.

**Rationale:** The metadata is human-readable and queryable without loading PyTorch. The
`CheckpointManager.list_checkpoints()` method can read metadata without loading tensors.

### Decision 4 — `TrainingResult` is a value object

`Trainer.train()` returns a `TrainingResult` dataclass and does NOT mutate the model
in-place after training (the model is updated during training, but the result object
captures the final state). The caller decides what to do with the trained model.

**Rationale:** This makes the trainer independently testable and compatible with the Flower
`ClientApp` pattern where the model update (not the model itself) is returned to the server.

### Decision 5 — Factory registry pattern

`ModelFactory` uses a string-keyed registry so new model architectures can be added
in future phases without modifying the factory.

**Rationale:** Phases 2–6 may need to test different architectures. The factory is the
single creation point — no scattered `if model_name == "resnet" ...` chains.

---

## Interfaces Frozen After Phase 1

These function signatures must not change without a documented breaking change:

```python
# ml/models/factory.py
ModelFactory.create(config: ModelConfig) -> nn.Module

# ml/training/trainer.py
Trainer.train(model, train_loader, val_loader, experiment_name) -> TrainingResult

# ml/training/evaluator.py
Evaluator.evaluate(model, loader, device) -> EvaluationResult

# ml/training/checkpoint.py
CheckpointManager.save(model, optimizer, metadata) -> Path
CheckpointManager.load(path) -> tuple[dict, dict, CheckpointMetadata]

# ml/datasets/cifar10.py
CIFAR10DataModule.train_loader -> DataLoader
CIFAR10DataModule.val_loader   -> DataLoader
CIFAR10DataModule.test_loader  -> DataLoader
```

---

## Dependencies for Phase 1

```toml
[project]
requires-python = ">=3.11"

[project.dependencies]
torch = ">=2.2.0,<3.0"
torchvision = ">=0.17.0,<1.0"
numpy = ">=1.26.0,<2.0"
pydantic = ">=2.6.0,<3.0"
pydantic-settings = ">=2.2.0,<3.0"
pyyaml = ">=6.0.1,<7.0"

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0,<9.0",
    "pytest-cov>=5.0.0,<6.0",
    "pytest-mock>=3.12.0,<4.0",
    "ruff>=0.4.0,<1.0",
    "mypy>=1.9.0,<2.0",
]
```

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| CIFAR-10 download fails in restricted network | Medium | Low | Cache check; allow pre-downloaded path |
| NaN loss on small synthetic test batches | Low | Low | Skip NaN detection in test mode, test with flag |
| Model too large to fit test memory | Low | Medium | Default is small CNN; configurable |
| Checkpoint path collision in parallel tests | Medium | Medium | `tmp_path` fixture isolates test directories |
| Phase 2 breaks Phase 1 interface | Medium | High | Interface contract documented; frozen after Phase 1 |
