# FedErase

**Resource-Adaptive Federated Unlearning Framework for Distributed Machine Learning**

FedErase is a research prototype that lets simulated federated learning clients train a
shared PyTorch model, tracks each client's contribution, and — when a client requests
deletion — estimates their influence and performs targeted approximate unlearning rather
than retraining the entire model from scratch.

---

## Scientific Position

> FedErase **estimates and mitigates** a target client's learned contribution and evaluates
> the resulting model against a reference model trained without that client.
>
> It does **not** guarantee exact removal of all knowledge from the model.
> Unlearning effectiveness is experimentally demonstrated, not mathematically proven.

This distinction is maintained throughout the codebase, all API responses, and the dashboard.

---

## Project Status

| Phase | Scope | Status |
|-------|-------|--------|
| **1** | **Centralized PyTorch baseline (CIFAR-10)** | **🔨 In progress** |
| 2 | Flower federated learning (IID) | Planned |
| 3 | Non-IID data partitioning | Planned |
| 4 | Client contribution tracking | Planned |
| 5 | Full-retraining reference baseline | Planned |
| 6 | FedErase v1 unlearning | Planned |
| 7 | Repair training | Planned |
| 8 | Resource-adaptive strategy selection | Planned |
| 9 | Real-time new-data adaptation | Planned |
| 10 | Model versioning and scheduler | Planned |
| 11 | FastAPI control plane | Planned |
| 12 | PostgreSQL persistence | Planned |
| 13 | React dashboard | Planned |
| 14 | Docker deployment | Planned |
| 15 | Full integration testing and benchmarking | Planned |

---

## Architecture (High-Level)

```
                     React Dashboard
                            │
                       FastAPI API
                            │
                   Federation Manager
                            │
                    Flower Federation
                   /        │        \
                 C1         C2        CN
                   \        │        /
                      FedAvg
                            │
                      Global Model
                            │
                  Contribution Engine
                            │
                      PostgreSQL
                            │
                    DELETE REQUEST
                            │
                   Unlearning Manager
                  /         │         \
         Influence    Resource      Strategy
         Estimator   Evaluator      Selector
                  \         │         /
                      Unlearning
                            │
                    Repair Training
                            │
                     Evaluation
                   /    │      \
              Accuracy Forgetting Cost
                            │
                     Model Registry
                            │
                    New Model Version
```

---

## Technology Stack

| Layer | Technology |
|-------|-----------|
| Core ML | Python 3.11+, PyTorch, TorchVision, NumPy |
| Federated Learning | Flower (ServerApp / ClientApp / simulation) |
| Dataset | CIFAR-10; FEMNIST/EMNIST (future) |
| Data partitioning | IID and Dirichlet non-IID |
| Backend API | FastAPI |
| Database | PostgreSQL (SQLite in Phase 1 only) |
| Frontend | React 18 + Vite + TypeScript |
| Charts | Recharts |
| Testing | pytest, pytest-cov, pytest-mock |
| Deployment | Docker + Docker Compose |

---

## Repository Structure

```
fed-erase/
├── README.md
├── pyproject.toml            # project metadata and all dependencies
├── .env.example              # environment variable documentation
├── Makefile                  # convenience targets
│
├── config/
│   └── defaults.yaml         # project-wide configuration defaults
│
├── experiments/
│   └── configs/
│       └── phase1_baseline.yaml
│
├── ml/                       # core ML — reused by all phases
│   ├── models/               # CNN, ModelFactory
│   ├── datasets/             # CIFAR-10 loader, partitioning, transforms
│   └── training/             # Trainer, Evaluator, CheckpointManager
│
├── apps/
│   ├── fl/                   # Flower ServerApp + ClientApp (Phase 2+)
│   ├── api/                  # FastAPI control plane (Phase 11+)
│   └── frontend/             # React dashboard (Phase 13+)
│
├── contribution/             # Contribution tracking (Phase 4+)
├── unlearning/               # Unlearning pipeline (Phase 6+)
├── evaluation/               # Evaluation engine (Phase 6+)
├── scheduler/                # Event bus + FL scheduler (Phase 9+)
├── database/                 # SQLAlchemy ORM + migrations (Phase 12+)
│
├── scripts/
│   └── run_phase1.py         # Phase 1 end-to-end training script
│
├── tests/                    # mirrors source tree
├── docker/                   # Dockerfiles (Phase 14+)
└── docs/
    ├── algorithm.md          # algorithmic assumptions and decisions
    ├── architecture.md       # system architecture reference
    ├── experiments.md        # experiment configurations and results
    └── api.md                # API reference (Phase 11+)
```

---

## Quick Start — Phase 1 (Centralized PyTorch)

Phase 1 trains a CNN on CIFAR-10 locally (no federation). This establishes and
verifies the ML core before any federated components are added.

### Prerequisites

- Python 3.11 or later
- pip
- Git
- (Optional) NVIDIA GPU with CUDA — the system falls back to CPU automatically

### 1. Clone the repository

```bash
git clone https://github.com/<your-org>/fed-erase.git
cd fed-erase
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Linux / macOS
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -e ".[dev]"
```

### 4. Configure environment

```bash
cp .env.example .env
# Edit .env if you need to override default paths
```

### 5. Run Phase 1 training

```bash
python scripts/run_phase1.py --config experiments/configs/phase1_baseline.yaml
```

This will:
1. Download and cache CIFAR-10 (≈170 MB, first run only).
2. Train a CNN for 20 epochs on CPU (or GPU if available).
3. Validate after each epoch and save the best checkpoint.
4. Print a summary table with validation accuracy, test accuracy, and checkpoint path.

**Expected output** (timings vary by hardware):
```
Phase 1 — Centralized CIFAR-10 Training
========================================
Experiment : phase1_baseline
Device     : cpu
Epochs     : 20
Seed       : 42

Epoch  1/20 | train_loss=2.142 | train_acc=0.231 | val_acc=0.318 | 48s
Epoch  2/20 | train_loss=1.783 | train_acc=0.352 | val_acc=0.409 | 47s
...
Epoch 20/20 | train_loss=0.921 | train_acc=0.681 | val_acc=0.643 | 46s

Best epoch       : 19
Best val accuracy: 64.7%
Test accuracy    : 63.2%
Checkpoint       : checkpoints/phase1/best.pt
Training time    : 15m 32s
```

> Note: Exact accuracy values depend on hardware, seed, and configuration.
> The system does not hard-code expected accuracy — results are always measured.

### 6. Run the test suite

```bash
# Unit and integration tests (fast)
make test

# Unit tests only
make test-unit

# With coverage report
make coverage

# Slow tests (includes a short training run — takes a few minutes)
pytest tests/ -m slow -v
```

### 7. Run linting and type checks

```bash
make lint
make typecheck
```

### 8. Other Makefile targets

```
make install     # reinstall dependencies
make format      # auto-format with ruff
make run-phase1  # same as step 5 above
```

---

## Configuration

All runtime parameters are in `experiments/configs/phase1_baseline.yaml`.
Override any value by editing the YAML — no code changes required.

Key parameters:

```yaml
seed: 42                    # reproducibility seed

dataset:
  batch_size: 64            # DataLoader batch size
  val_fraction: 0.1         # fraction of training set used for validation
  augment_train: true       # random crop + horizontal flip on training split

model:
  conv_channels: [32, 64, 128]   # filter counts per conv block
  dropout: 0.3

training:
  epochs: 20
  optimizer: sgd            # sgd | adam
  learning_rate: 0.01
  scheduler: cosine         # cosine | step | null
  early_stopping_patience: 7
  device: auto              # auto | cpu | cuda
```

For a quick 2-epoch smoke test, create a copy with `epochs: 2`:
```bash
cp experiments/configs/phase1_baseline.yaml experiments/configs/phase1_quick.yaml
# Edit phase1_quick.yaml: set epochs: 2
python scripts/run_phase1.py --config experiments/configs/phase1_quick.yaml
```

---

## Checkpoints

Checkpoints are saved to `checkpoints/phase1/` (configured via `checkpoint.checkpoint_dir`).

Each checkpoint consists of two files:
- `best.pt` — PyTorch state dict (model + optimizer)
- `best.meta.json` — metadata including epoch, accuracy, config hash, and SHA-256 checksum

The checksum is verified on every load. A corrupted checkpoint raises an error — it is
never silently ignored.

Checkpoints and the `data/` directory are excluded from git.

---

## Phase 1 Acceptance Criteria

Phase 1 is complete when all of the following are true:

| # | Criterion |
|---|-----------|
| AC-1 | CIFAR-10 downloads and caches on first run |
| AC-2 | Second run uses cache, does not re-download |
| AC-3 | Model forward pass produces shape `(B, 10)` |
| AC-4 | ModelFactory creates model from config dict |
| AC-5 | Training runs 5 epochs without error |
| AC-6 | Per-epoch loss and accuracy are logged |
| AC-7 | Validation runs after each epoch |
| AC-8 | NaN loss detection halts training with a WARNING |
| AC-9 | Evaluator returns correct accuracy on known data |
| AC-10 | Checkpoint is saved after training |
| AC-11 | Checkpoint includes SHA-256 checksum |
| AC-12 | Checkpoint loads and model produces same output |
| AC-13 | Checksum mismatch raises `ChecksumMismatchError` |
| AC-14 | Configuration loads and validates from YAML |
| AC-15 | Invalid config raises `ValidationError` with clear message |
| AC-16 | Same seed produces reproducible results |
| AC-17 | All unit tests pass |
| AC-18 | All integration tests pass |
| AC-19 | `ml/` has no imports from other project packages |
| AC-20 | End-to-end 5-epoch run produces a valid checkpoint |

---

## Specification Documents

Detailed specs for Phase 1 are in `.kiro/specs/phase1-centralized-pytorch/`:

| Document | Contents |
|----------|----------|
| `requirements.md` | Functional requirements, non-functional requirements, acceptance criteria |
| `design.md` | Component designs, class signatures, data flow, design decisions |
| `tasks.md` | 13 ordered implementation tasks with step-by-step instructions and test checklists |

Project-wide steering rules are in `.kiro/steering/`:

| Document | Contents |
|----------|----------|
| `project-overview.md` | Research goals, phase table, tech stack |
| `architecture.md` | Component map, module responsibilities, dependency rules |
| `coding-standards.md` | Python/TS conventions, logging, error handling |
| `ml-research-rules.md` | 11 rules governing scientific claims and algorithmic decisions |
| `testing-rules.md` | Test categories, coverage targets, edge cases |
| `security-rules.md` | 10 privacy and security rules for the prototype |

---

## Research Comparisons (Phase 6+)

Once unlearning is implemented, the system will compare three approaches on the same
experiment configuration:

| Approach | Description |
|----------|-------------|
| Full Retraining | Remove target client, retrain from scratch — ground truth |
| FedErase Baseline | Contribution correction + repair training |
| Resource-Adaptive FedErase | Strategy selected based on influence score and available resources |

Metrics compared: accuracy, forgetting score, remaining-client utility, computation time,
communication cost, storage, unlearning latency.

Results are generated by the experiment system — never hard-coded.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) *(coming soon)*.

Branch naming: `phase/<n>-<short-description>` — e.g. `phase/1-centralized-pytorch`.

Commit format: `<type>(<scope>): <subject>` — e.g. `feat(ml): add CNN model factory`.

---

## License

[MIT License](LICENSE) *(to be added)*

---

## Known Limitations (Phase 1)

- No federation — this is a single-machine, centralized training loop only.
- No contribution tracking — that is Phase 4.
- No unlearning — that is Phase 6.
- No API or dashboard — those are Phases 11 and 13.
- SQLite is not yet needed — Phase 1 is filesystem-only.
- The membership-inference evaluator and differential privacy mechanisms are not implemented.
- Resource profiles are simulated configuration values, not actual hardware measurements.
- This prototype uses CIFAR-10 only. Do not use it with real sensitive data without
  a separate privacy and security review.
