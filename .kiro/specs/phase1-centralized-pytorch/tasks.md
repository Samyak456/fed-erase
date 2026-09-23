# Phase 1 — Centralized PyTorch Baseline
## Implementation Tasks

**Phase:** 1 of 15  
**Status:** Ready for implementation  
**Prerequisite reading:** `requirements.md`, `design.md`

Complete tasks in the order listed. Each task has a clear deliverable and a set of
acceptance sub-checks. Do not mark a task complete unless all its sub-checks pass.

---

## Task 1 — Project Skeleton

**Deliverable:** Repository structure, `pyproject.toml`, `.env.example`, `.gitignore`, `Makefile`.

### Steps

1.1 Create directory tree:
```
fed-erase/
├── config/
├── experiments/configs/
├── ml/models/
├── ml/datasets/
├── ml/training/
├── scripts/
├── tests/ml/datasets/
├── tests/ml/models/
├── tests/ml/training/
├── docs/
├── .kiro/steering/         (already exists)
└── .kiro/specs/            (already exists)
```

1.2 Create `pyproject.toml` with:
- Project name `fed-erase`, version `0.1.0`
- Python `>=3.11` requirement
- All Phase 1 runtime dependencies (pinned exact minor versions)
- Dev dependency group: pytest, pytest-cov, pytest-mock, ruff, mypy
- `[tool.ruff]` configuration (line length 100, enabled rule sets)
- `[tool.pytest.ini_options]` with `testpaths = ["tests"]` and marker definitions
- `[tool.mypy]` baseline configuration

1.3 Create `.env.example` documenting all environment variables used in Phase 1.

1.4 Create `.gitignore` covering: `__pycache__`, `.env`, `*.pt`, `*.pth`, `data/`,
`checkpoints/`, `.venv/`, `dist/`, `*.egg-info/`, `node_modules/`, `.pytest_cache/`,
`htmlcov/`, `*.log`.

1.5 Create `Makefile` with targets:
- `install` — `pip install -e ".[dev]"`
- `test` — `pytest tests/ -m "unit or integration" -v`
- `test-unit` — `pytest tests/ -m unit -v`
- `test-slow` — `pytest tests/ -m slow -v`
- `lint` — `ruff check .`
- `format` — `ruff format .`
- `typecheck` — `mypy ml/`
- `run-phase1` — `python scripts/run_phase1.py --config experiments/configs/phase1_baseline.yaml`
- `coverage` — `pytest --cov=ml --cov-report=term-missing --cov-report=html`

1.6 Add `__init__.py` to all package directories.

### Acceptance checks
- [ ] `pip install -e ".[dev]"` succeeds without errors.
- [ ] `make lint` runs (may have warnings on empty files, must not crash).
- [ ] `make test` runs (0 tests collected is acceptable at this stage).
- [ ] Directory tree matches the spec exactly.

---

## Task 2 — Configuration Schema

**Deliverable:** `ml/config.py` (or `config/schema.py`) with Pydantic config schema and loader.

### Steps

2.1 Implement the full `FedEraseConfig` Pydantic model as specified in `design.md §1`.

2.2 Implement `load_config(path: Path) -> FedEraseConfig`:
- Reads YAML file.
- Validates with Pydantic (fails fast on invalid config with a clear `ValidationError`).
- Resolves relative `Path` fields against the project root.
- Calls `set_seeds(config.seed)` before returning.
- Logs `CONFIG_LOADED` at INFO level.

2.3 Implement `set_seeds(seed: int) -> None`:
- Sets `random.seed`, `numpy.random.seed`, `torch.manual_seed`, and
  `torch.cuda.manual_seed_all` (if CUDA available).
- Logs the seed at DEBUG level.

2.4 Implement `compute_config_hash(path: Path) -> str`:
- Returns SHA-256 of the YAML file bytes (for checkpoint metadata).

2.5 Create `experiments/configs/phase1_baseline.yaml` with sensible defaults:
```yaml
experiment_name: phase1_baseline
seed: 42

dataset:
  name: cifar10
  data_dir: data/cifar10
  val_fraction: 0.1
  batch_size: 64
  num_workers: 2
  augment_train: true

model:
  name: cifar_cnn
  num_classes: 10
  conv_channels: [32, 64, 128]
  dropout: 0.3
  activation: relu

training:
  epochs: 20
  optimizer: sgd
  learning_rate: 0.01
  momentum: 0.9
  weight_decay: 0.0001
  scheduler: cosine
  early_stopping_patience: 7
  device: auto

checkpoint:
  checkpoint_dir: checkpoints/phase1
  save_best_only: true
  keep_last_n: 3

logging:
  level: INFO
  log_dir: logs
  log_file: phase1.log
```

2.6 Create `config/defaults.yaml` as a copy of the above (future phases will
override from experiment configs).

### Tests (`tests/ml/test_config.py`)
- [ ] `load_config` succeeds on valid YAML.
- [ ] `load_config` raises `ValidationError` on missing required field.
- [ ] `load_config` raises `ValidationError` on wrong type (e.g. `epochs: "twenty"`).
- [ ] `load_config` raises `FileNotFoundError` on missing file.
- [ ] `set_seeds` does not raise for seed=0 or seed=99999.
- [ ] `compute_config_hash` returns a 64-character hex string.
- [ ] Same config file produces same hash across calls.

### Acceptance checks
- [ ] All config tests pass.
- [ ] `phase1_baseline.yaml` loads without error.
- [ ] Invalid YAML produces a human-readable error message.

---

## Task 3 — Data Transforms

**Deliverable:** `ml/datasets/transforms.py`

### Steps

3.1 Define `CIFAR10_MEAN` and `CIFAR10_STD` as module-level constants.

3.2 Implement `get_train_transforms(augment: bool) -> transforms.Compose`:
- If `augment=True`: RandomCrop(32, padding=4) → RandomHorizontalFlip → ToTensor → Normalize.
- If `augment=False`: ToTensor → Normalize only.

3.3 Implement `get_eval_transforms() -> transforms.Compose`:
- ToTensor → Normalize only.

3.4 Add module docstring explaining that these are CIFAR-10-specific constants
taken from standard dataset statistics.

### Tests (`tests/ml/datasets/test_transforms.py`)
- [ ] `get_train_transforms(augment=True)` returns a `Compose` with 4 transforms.
- [ ] `get_train_transforms(augment=False)` returns a `Compose` with 2 transforms.
- [ ] `get_eval_transforms()` returns a `Compose` with 2 transforms.
- [ ] Applying eval transform to a PIL image returns a tensor of shape `(3, 32, 32)`.
- [ ] Normalised pixel values are in roughly `[-3, 3]` range.

### Acceptance checks
- [ ] All transform tests pass.

---

## Task 4 — CIFAR-10 Dataset Loader

**Deliverable:** `ml/datasets/cifar10.py`

### Steps

4.1 Implement `CIFAR10DataModule` class as specified in `design.md §2`.

4.2 `prepare_data()`:
- Creates `data_dir` if it does not exist.
- Downloads CIFAR-10 to `data_dir` using `torchvision.datasets.CIFAR10`.
- Logs `DATASET_DOWNLOAD_STARTED` and `DATASET_DOWNLOAD_COMPLETED` (or
  `DATASET_CACHE_HIT` if already present).
- Must be idempotent — calling twice does not re-download.

4.3 `setup()`:
- Loads the training set with train transforms (augment based on config).
- Loads the test set with eval transforms.
- Creates train/val indices: shuffle the 50 000 training indices using the configured
  seed, then split at `floor(50000 * (1 - val_fraction))`.
- Creates `Subset` objects for train and val.
- Creates `DataLoader` objects for all three splits.

4.4 Expose `train_loader`, `val_loader`, `test_loader` as properties that raise
`RuntimeError` with a clear message if `setup()` has not been called.

4.5 Expose `num_classes: int = 10` and `class_names: list[str]` as properties.

4.6 Add a `__main__` block for standalone invocation:
```python
if __name__ == "__main__":
    # Quick sanity check: print dataset stats
```

### Tests (`tests/ml/datasets/test_cifar10.py`)

Use a synthetic dataset fixture (see `tests/conftest.py`) — do NOT require a real CIFAR-10
download in unit tests. Mark real-download tests as `@pytest.mark.integration`.

- [ ] `prepare_data()` creates the data directory if missing (unit, mocked download).
- [ ] `prepare_data()` is idempotent — second call does not re-download (unit).
- [ ] After `setup()`, `train_loader` is not None.
- [ ] After `setup()`, `val_loader` is not None.
- [ ] After `setup()`, `test_loader` is not None.
- [ ] `train_loader` + `val_loader` indices are disjoint (unit, synthetic data).
- [ ] Total train + val samples == 50 000 (integration).
- [ ] Test set has exactly 10 000 samples (integration).
- [ ] Batch shape from `train_loader` is `(B, 3, 32, 32)` (unit, synthetic data).
- [ ] Labels are integers in `[0, 9]` (unit, synthetic data).
- [ ] Accessing `train_loader` before `setup()` raises `RuntimeError`.
- [ ] Same seed produces same train/val split (unit, synthetic data).
- [ ] Different seeds produce different splits (unit, synthetic data).
- [ ] `num_classes == 10` (unit).
- [ ] `len(class_names) == 10` (unit).

### Acceptance checks
- [ ] All unit tests pass.
- [ ] Integration test (real download) passes when run with `make test-slow`.

---

## Task 5 — CNN Architecture

**Deliverable:** `ml/models/cnn.py`

### Steps

5.1 Implement `CifarCNN(nn.Module)` with:
- Constructor accepting a `ModelConfig` (or equivalent config dataclass).
- Dynamically builds conv blocks from `config.conv_channels`.
- Computes FC input dimension from a dummy forward pass on an input of shape `(1, 3, 32, 32)`.
- Uses BatchNorm after each conv layer.
- Uses MaxPool(2×2) after each conv block.
- Uses Dropout before the final FC layer.
- Supports `relu` and `gelu` as activation functions (controlled by `config.activation`).
- Raises `ValueError` for unsupported activation names.

5.2 The `forward(x: Tensor) -> Tensor` method must:
- Accept input shape `(B, 3, 32, 32)`.
- Return output shape `(B, num_classes)` (raw logits).

5.3 Add a `parameter_count() -> int` method returning the total number of trainable parameters.

5.4 Add a `__repr__` that includes the architecture summary (number of layers, parameter count).

### Tests (`tests/ml/models/test_cnn.py`)
- [ ] Instantiation succeeds with default config.
- [ ] `forward()` output shape is `(4, 10)` for batch size 4.
- [ ] Output is finite (no NaN or Inf in a randomly initialised forward pass).
- [ ] `parameter_count()` returns a positive integer.
- [ ] `state_dict()` can be serialised and deserialised (round-trip).
- [ ] After loading a saved state_dict, forward pass produces identical output.
- [ ] `conv_channels=[64, 128]` (2 blocks) instantiates correctly.
- [ ] `conv_channels=[32, 64, 128, 256]` (4 blocks) instantiates correctly.
- [ ] `activation="gelu"` instantiates correctly.
- [ ] `activation="tanh"` raises `ValueError`.

### Acceptance checks
- [ ] All model tests pass.

---

## Task 6 — Model Factory

**Deliverable:** `ml/models/factory.py`

### Steps

6.1 Implement `ModelFactory` with:
- `_REGISTRY: dict[str, type[nn.Module]]` mapping `"cifar_cnn"` to `CifarCNN`.
- `create(config: ModelConfig) -> nn.Module` — looks up registry, constructs model.
- `register(name: str, cls: type[nn.Module]) -> None` — adds to registry.

6.2 `create` must raise `ValueError` with a message listing available model names if
`config.name` is not found.

### Tests (add to `tests/ml/models/test_cnn.py`)
- [ ] `ModelFactory.create` with `name="cifar_cnn"` returns a `CifarCNN` instance.
- [ ] `ModelFactory.create` with unknown name raises `ValueError` containing the name.
- [ ] `ModelFactory.register` adds a new model and `create` uses it.

### Acceptance checks
- [ ] Factory tests pass.

---

## Task 7 — Evaluator

**Deliverable:** `ml/training/evaluator.py`

### Steps

7.1 Implement `EvaluationResult` dataclass.

7.2 Implement `Evaluator` class:
- Constructor accepts optional `criterion: nn.Module` (defaults to `nn.CrossEntropyLoss()`).
- `evaluate(model, loader, device)` method:
  - Sets model to `eval()` mode.
  - Runs all batches under `torch.no_grad()`.
  - Accumulates correct predictions and total loss.
  - Restores model to `train()` mode before returning (caller owns the mode).
  - Records start/end time for `duration_seconds`.
  - Logs `EVALUATION_COMPLETED` at INFO level.

### Tests (`tests/ml/training/test_evaluator.py`)

Use synthetic DataLoader with known labels for deterministic tests.

- [ ] `evaluate()` returns `EvaluationResult` with correct fields.
- [ ] `accuracy` is in `[0.0, 1.0]`.
- [ ] `loss` is non-negative.
- [ ] `n_samples` equals the total number of samples in the DataLoader.
- [ ] Model weights are unchanged after evaluation (compare state_dict before/after).
- [ ] `evaluate()` handles a DataLoader with a single batch of size 1.
- [ ] `evaluate()` handles a perfect classifier (all predictions correct → accuracy=1.0).
- [ ] `evaluate()` handles a random initialised model on a 1-class problem.

### Acceptance checks
- [ ] All evaluator tests pass.

---

## Task 8 — Checkpoint Manager

**Deliverable:** `ml/training/checkpoint.py`

### Steps

8.1 Define `ChecksumMismatchError(Exception)` in `ml/exceptions.py`.

8.2 Implement `CheckpointMetadata` dataclass.

8.3 Implement `CheckpointManager`:
- `__init__(config: CheckpointConfig, experiment_name: str)` — creates `checkpoint_dir` if needed.
- `save(model, optimizer, metadata) -> Path`:
  - Writes `<experiment>_epoch<n>.pt` using `torch.save`.
  - Raises if that exact path already exists.
  - Computes SHA-256 of the written file.
  - Writes `<filename>.meta.json` with metadata + checksum.
  - Logs `CHECKPOINT_SAVED`.
  - Returns the `.pt` path.
- `save_best(model, optimizer, metadata) -> Path`:
  - Renames the previous `best.pt` to `best_prev.pt` before writing (never overwrites).
  - Writes `best.pt` and `best.meta.json`.
  - Logs `CHECKPOINT_SAVED` with `best=True`.
- `load(path) -> tuple[dict, dict, CheckpointMetadata]`:
  - Reads the corresponding `.meta.json` for the checksum.
  - Re-computes SHA-256 of the `.pt` file.
  - Raises `ChecksumMismatchError` if they differ.
  - Loads the checkpoint with `torch.load(..., map_location="cpu")`.
  - Logs `CHECKPOINT_LOADED`.
  - Returns `(state_dict, optimizer_state, metadata)`.
- `list_checkpoints() -> list[CheckpointMetadata]` — reads all `.meta.json` files.
- `cleanup(keep_last_n)` — deletes oldest `.pt` + `.meta.json` pairs, keeping N newest
  plus the `best.*` pair.

8.4 The `.meta.json` format:
```json
{
  "model_version": "phase1_baseline_v1",
  "experiment_name": "phase1_baseline",
  "epoch": 5,
  "val_accuracy": 0.6312,
  "train_loss": 0.8741,
  "config_hash": "abc123...",
  "created_at": "2026-09-20T12:00:00Z",
  "checksum": "sha256:def456...",
  "path": "checkpoints/phase1/phase1_baseline_epoch5.pt"
}
```

### Tests (`tests/ml/training/test_checkpoint.py`)

All tests use `tmp_path` fixture for isolation.

- [ ] `save()` creates both `.pt` and `.meta.json` files.
- [ ] `save()` raises if the same path already exists.
- [ ] `save()` stores a non-empty SHA-256 checksum in the metadata.
- [ ] `load()` returns a state_dict that matches the saved model's `state_dict()`.
- [ ] `load()` + forward pass produces identical output to the original model.
- [ ] `load()` raises `FileNotFoundError` for missing `.pt` file.
- [ ] `load()` raises `ChecksumMismatchError` when the `.pt` file is modified after saving.
- [ ] `load()` raises when the `.meta.json` is missing.
- [ ] `save_best()` renames previous best before writing new best.
- [ ] `list_checkpoints()` returns correct count after multiple saves.
- [ ] `cleanup(keep_last_n=2)` deletes oldest, keeps 2 + best.
- [ ] Metadata fields are preserved through save/load round-trip.

### Acceptance checks
- [ ] All checkpoint tests pass.

---

## Task 9 — Trainer

**Deliverable:** `ml/training/trainer.py`

### Steps

9.1 Implement `EpochRecord` and `TrainingResult` dataclasses.

9.2 Implement helper `_resolve_device(device_config: str) -> torch.device`:
- `"auto"` → CUDA if available, else CPU.
- `"cpu"` → CPU.
- `"cuda"` → CUDA (raises if not available).
- Logs device selection at INFO.

9.3 Implement `_build_optimizer(model, config: TrainingConfig) -> Optimizer`:
- `"sgd"` → `torch.optim.SGD(lr, momentum, weight_decay)`.
- `"adam"` → `torch.optim.Adam(lr, weight_decay)`.
- Raises `ValueError` for unknown optimizer names.

9.4 Implement `_build_scheduler(optimizer, config: TrainingConfig)`:
- `"cosine"` → `CosineAnnealingLR(T_max=config.epochs)`.
- `"step"` → `StepLR(step_size=5, gamma=0.5)`.
- `None` → no scheduler.
- Raises `ValueError` for unknown scheduler names.

9.5 Implement `Trainer`:
- Constructor stores config and checkpoint_manager.
- `train(model, train_loader, val_loader, experiment_name) -> TrainingResult`:
  - Follows the pseudocode in `design.md §5`.
  - Uses the internal `Evaluator` for validation each epoch.
  - Calls `checkpoint_manager.save_best(...)` on improvement.
  - Returns full `TrainingResult`.

### Tests (`tests/ml/training/test_trainer.py`)

Use a small synthetic DataLoader (e.g. 4 batches of size 8 with random tensors).

- [ ] `train()` completes 3 epochs on synthetic data without error.
- [ ] `TrainingResult` has all required fields.
- [ ] `history` has one `EpochRecord` per completed epoch.
- [ ] `best_val_accuracy` is in `[0.0, 1.0]`.
- [ ] NaN loss triggers `WARNING` and stops training early.
- [ ] `early_stopping_patience=1` stops after 2 non-improving epochs.
- [ ] Device is CPU when `config.device="cpu"`.
- [ ] Checkpoint is called on validation improvement.
- [ ] `epochs=0` returns a `TrainingResult` with `total_epochs=0` (does not crash).
- [ ] Optimizer `"sgd"` builds without error.
- [ ] Optimizer `"adam"` builds without error.
- [ ] Optimizer `"invalid"` raises `ValueError`.

### Acceptance checks
- [ ] All trainer tests pass.

---

## Task 10 — Partitioning Stub

**Deliverable:** `ml/datasets/partitioning.py` (Phase 1 stub, fully implemented in Phase 3)

### Steps

10.1 Define `PartitionConfig` dataclass:
```python
@dataclass
class PartitionConfig:
    strategy: str = "iid"   # "iid" | "dirichlet"
    num_clients: int = 10
    alpha: float = 0.5       # only used for dirichlet
    seed: int = 42
```

10.2 Implement `iid_partition(dataset, num_clients, seed) -> list[Subset]`:
- Divides dataset indices into `num_clients` roughly equal-sized subsets.
- Shuffles indices using `seed` before splitting.

10.3 Implement `dirichlet_partition(dataset, num_clients, alpha, seed) -> list[Subset]`:
- Phase 1 stub — raises `NotImplementedError` with message:
  `"Dirichlet partitioning is implemented in Phase 3."`

10.4 Implement `create_partitions(dataset, config: PartitionConfig) -> list[Subset]`:
- Dispatches to `iid_partition` or `dirichlet_partition` based on `config.strategy`.

### Tests (`tests/ml/datasets/test_partitioning.py`)
- [ ] IID partition produces `num_clients` subsets.
- [ ] Total samples across all IID subsets equals the full dataset size.
- [ ] No sample index appears in more than one IID subset.
- [ ] Dirichlet partition raises `NotImplementedError` in Phase 1.
- [ ] `create_partitions` with `strategy="iid"` calls `iid_partition`.
- [ ] `create_partitions` with `strategy="dirichlet"` raises `NotImplementedError`.
- [ ] IID is reproducible with the same seed.
- [ ] IID with different seeds produces different orderings.

### Acceptance checks
- [ ] All partitioning tests pass.

---

## Task 11 — Entry Point Script and docs

**Deliverable:** `scripts/run_phase1.py`, `docs/algorithm.md`, updated `README.md`

### Steps

11.1 Implement `scripts/run_phase1.py` per the design:
- Argument: `--config` (required).
- Full pipeline: config → data → model → trainer → evaluator → checkpoint → summary.
- Prints a summary table: val accuracy, test accuracy, best epoch, total time, checkpoint path.
- Exits with code 0 on success, 1 on any unhandled error.

11.2 Create `docs/algorithm.md` with:
- Section 1: Centralized CIFAR-10 baseline — architecture description, design decisions
  from `design.md`, limitations.
- Section 2 placeholder: Federated Learning (Phase 2) — TBD.
- Sections 3–N: Placeholder headings for each future phase.

11.3 Update `README.md` with Phase 1 content (see Task 12).

### Tests
- [ ] `scripts/run_phase1.py --config experiments/configs/phase1_baseline.yaml` exits with 0
  after a short test run (use `epochs: 2` override for the test).
- [ ] Script exits with 1 and a clear error when given a non-existent config path.

### Acceptance checks
- [ ] Script runs end-to-end.
- [ ] `docs/algorithm.md` exists with Phase 1 content.

---

## Task 12 — README

**Deliverable:** `README.md` (root of repository)

### Content

The README must contain:

1. **Title and one-sentence description.**
2. **Scientific position statement** (exact language from `ml-research-rules.md`).
3. **Phase progress table** (Phase 1: complete; Phases 2–15: planned).
4. **Architecture diagram** (ASCII, high-level).
5. **Quick start — Phase 1:**
   - Prerequisites (Python 3.11, pip).
   - Installation (`git clone`, `pip install -e ".[dev]"`).
   - Configuration (copy `.env.example` to `.env`).
   - Run centralized training.
   - Run tests.
6. **Directory structure** (top-level only).
7. **Technology stack table.**
8. **Contributing** pointer to `CONTRIBUTING.md` (placeholder).
9. **License** pointer.

### Acceptance checks
- [ ] README renders correctly in a Markdown viewer.
- [ ] All commands in the Quick Start section are accurate and executable.

---

## Task 13 — Hooks and Kiro Configuration

**Deliverable:** `.kiro/hooks/` — useful development hooks.

### Steps

13.1 Create a `PostFileSave` hook for Python files:
- Trigger: save of any `.py` file.
- Action: run `ruff check` on the saved file and report errors.

13.2 Create a `PostFileSave` hook for test files:
- Trigger: save of any file matching `tests/**/*.py`.
- Action: run pytest on the specific test file.

### Acceptance checks
- [ ] Hooks are valid JSON.
- [ ] Hooks fire correctly on file save.

---

## Final Phase 1 Checklist

Before declaring Phase 1 complete, verify every item:

### Code
- [ ] All source files have module-level docstrings.
- [ ] All public functions have docstrings with Args/Returns/Raises.
- [ ] All type annotations are present on public interfaces.
- [ ] `ruff check .` reports no errors.
- [ ] `mypy ml/` reports no errors.

### Tests
- [ ] `pytest tests/ -m unit` — all pass, 0 failures.
- [ ] `pytest tests/ -m integration` — all pass, 0 failures.
- [ ] Coverage on `ml/` is ≥ 85%.
- [ ] No test depends on execution order.
- [ ] No test requires CIFAR-10 download unless marked `@pytest.mark.integration`.

### Configuration
- [ ] `phase1_baseline.yaml` loads without errors.
- [ ] All configurable values are in the YAML — none hard-coded.
- [ ] `.env.example` documents all environment variables.

### Functional
- [ ] `python scripts/run_phase1.py --config experiments/configs/phase1_baseline.yaml`
  completes a 5-epoch run.
- [ ] A checkpoint is written to `checkpoints/phase1/`.
- [ ] Checkpoint loads and produces the same output as the saved model.
- [ ] Test accuracy is reported.

### Documentation
- [ ] `README.md` is complete with Phase 1 quick start.
- [ ] `docs/algorithm.md` has Phase 1 section.
- [ ] Phase 1 design decisions are documented.

### Safety
- [ ] No credentials in source code.
- [ ] `checkpoints/` and `data/` are in `.gitignore`.
- [ ] No ML computation in API code (there is no API yet, but the pattern is set).

---

## Reporting Template

When Phase 1 implementation is complete, report using this template:

### Changed files
[List all created/modified files]

### What was implemented
[2–5 bullet points]

### Tests executed
[pytest command and output summary]

### Test results
[Pass count, fail count, coverage]

### How to run it
[Exact commands]

### Known limitations
[Honest list of what Phase 1 does not do]

### Next recommended task
Phase 2 — Flower Federated Learning
