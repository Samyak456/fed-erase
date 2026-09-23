# FedErase — System Architecture

## Architectural Principles

1. **Separation of concerns.** ML logic, federation logic, unlearning logic, API logic, and persistence are in separate modules. No module does more than one job.
2. **Incremental build.** Each phase produces a working, testable system. Never skip a phase.
3. **Configurability over hard-coding.** Every threshold, hyperparameter, strategy weight, and dataset setting is in a configuration file or environment variable.
4. **Scientific transparency.** Every algorithmic assumption is documented. Nothing is silently approximated.
5. **Reusability.** The centralized PyTorch model (Phase 1) must be usable inside Flower clients (Phase 2+) without modification.

---

## Component Map

```
┌─────────────────────────────────────────────────────────────────────┐
│                          React Dashboard                            │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ HTTP / WebSocket
┌──────────────────────────────▼──────────────────────────────────────┐
│                         FastAPI (apps/api/)                         │
│  Routes: /federation  /clients  /unlearning  /models  /metrics      │
│  Services: FederationService  UnlearningService  ModelRegistryService│
└──────────────────────────────┬──────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│                    Federation Manager                               │
│  Orchestrates FL rounds, client selection, model aggregation        │
│  Delegates to Flower ServerApp for the actual federation protocol   │
└──────┬────────────────────┬─────────────────────┬───────────────────┘
       │                    │                     │
       ▼                    ▼                     ▼
  Flower ServerApp    PostgreSQL DB         FL Scheduler
  (apps/fl/)          (database/)           (scheduler/)
       │
  ┌────┴─────────────────────────────────┐
  │         Flower Simulation Runtime    │
  │  ClientApp × N (apps/fl/client_app)  │
  │  Each client has:                    │
  │   - local dataset partition          │
  │   - local model state                │
  │   - resource profile                 │
  │   - data buffer (Phase 9+)           │
  └────┬─────────────────────────────────┘
       │ local updates (weights / grads)
       ▼
  FedAvg Aggregation
       │
       ▼
  Global Model (ml/models/)
       │
       ▼
  Contribution Engine (contribution/)
  - ContributionTracker
  - FeatureExtractor
  - InfluenceEstimator
  - ContributionStorage → PostgreSQL
       │
       │  ← DELETE REQUEST
       ▼
  Unlearning Manager (unlearning/)
  - InfluenceEstimator
  - ResourceProfiler / ResourceEvaluator
  - StrategySelector
  - UnlearningStrategy (Lightweight / Selective / Strong)
  - RepairTrainer
  - UnlearningPipeline
       │
       ▼
  Evaluation Engine (evaluation/)
  - AccuracyEvaluator
  - ForgettingEvaluator
  - PrivacyEvaluator (optional)
  - EfficiencyEvaluator
  - ComparisonReporter
       │
       ▼
  Model Registry (apps/api/services/model_registry_service.py)
  - Versioned checkpoints
  - Metadata in PostgreSQL
       │
       ▼
  New Model Version → React Dashboard
```

---

## Module Responsibilities

### `ml/` — Core Machine Learning

| Module | Responsibility |
|--------|---------------|
| `ml/models/cnn.py` | CNN architecture definition |
| `ml/models/factory.py` | Create model instances by name/config |
| `ml/datasets/cifar10.py` | Download, cache, and return CIFAR-10 splits |
| `ml/datasets/partitioning.py` | IID and Dirichlet non-IID partitioning |
| `ml/datasets/transforms.py` | Data augmentation and normalization transforms |
| `ml/training/trainer.py` | Training loop (epoch loop, loss, optimizer step) |
| `ml/training/evaluator.py` | Validation and test evaluation |
| `ml/training/checkpoint.py` | Save, load, version, and checksum model states |

**Phase 1 scope:** all of `ml/`. No federation, no unlearning.

### `apps/fl/` — Flower Federation Layer

| Module | Responsibility |
|--------|---------------|
| `apps/fl/server_app.py` | Flower ServerApp, strategy wiring |
| `apps/fl/client_app.py` | Flower ClientApp, local training per client |
| `apps/fl/strategy.py` | FedAvg extension hooks (contribution capture, etc.) |
| `apps/fl/config.py` | FL hyperparameters, simulation config |

**Phase 2+ scope.**

### `contribution/` — Contribution Engine

| Module | Responsibility |
|--------|---------------|
| `contribution/tracker.py` | Per-round contribution record assembly |
| `contribution/features.py` | Feature extraction (norms, cosine sim, layer stats) |
| `contribution/influence.py` | Composite influence score computation |
| `contribution/storage.py` | Persist contribution records to database |

**Phase 4+ scope.**

### `unlearning/` — Unlearning Pipeline

| Module | Responsibility |
|--------|---------------|
| `unlearning/baseline/retraining.py` | Full retraining reference (Phase 5) |
| `unlearning/strategies/lightweight.py` | Low-influence, low-resource strategy |
| `unlearning/strategies/selective.py` | Layer-targeted strategy |
| `unlearning/strategies/strong.py` | High-influence strategy |
| `unlearning/influence.py` | Unlearning-side influence estimation |
| `unlearning/adaptive.py` | ResourceProfiler, ResourceEvaluator, StrategySelector |
| `unlearning/repair.py` | Repair training loop |
| `unlearning/pipeline.py` | End-to-end unlearning orchestration |

**Phase 6+ scope.**

### `evaluation/` — Evaluation Engine

| Module | Responsibility |
|--------|---------------|
| `evaluation/accuracy.py` | Global and per-client accuracy / loss |
| `evaluation/forgetting.py` | Parameter distance, prediction disagreement, KL div |
| `evaluation/privacy.py` | Membership-inference-style evaluation (optional) |
| `evaluation/efficiency.py` | Latency, CPU/GPU time, communication, storage |
| `evaluation/comparison.py` | Side-by-side FedErase vs. full-retraining report |

**Phase 6+ scope.**

### `scheduler/` — Event and Scheduling Layer

| Module | Responsibility |
|--------|---------------|
| `scheduler/events.py` | Event type definitions and in-process event bus |
| `scheduler/scheduler.py` | FL round scheduler, eligibility evaluation |
| `scheduler/policies.py` | Configurable eligibility and scheduling policies |

**Phase 9+ scope.** Interfaces must be designed for later broker replacement.

### `database/` — Persistence

| Module | Responsibility |
|--------|---------------|
| `database/models.py` | SQLAlchemy ORM table definitions |
| `database/session.py` | Session factory, connection management |
| `database/migrations/` | Alembic migration scripts |

**Phase 12 scope.** SQLite allowed in Phase 1 integration.

### `apps/api/` — FastAPI Control Plane

**Phase 11+ scope.**

### `apps/frontend/` — React Dashboard

**Phase 13+ scope.**

---

## Data Flow: Phase 1 (Centralized)

```
config.yaml
    │
    ▼
ConfigLoader
    │
    ├──► DatasetLoader (ml/datasets/cifar10.py)
    │         │
    │         ▼
    │    train / val / test DataLoaders
    │
    ├──► ModelFactory (ml/models/factory.py)
    │         │
    │         ▼
    │      CNN model
    │
    └──► Trainer (ml/training/trainer.py)
              │
              ├── epoch loop
              │      └── optimizer.step()
              │
              ├──► Evaluator (ml/training/evaluator.py)
              │         └── val loss / accuracy per epoch
              │
              └──► CheckpointManager (ml/training/checkpoint.py)
                        └── saves best model + metadata
```

---

## Configuration Hierarchy

```
experiments/configs/<experiment>.yaml   ← experiment-level overrides
    └── inherits from
        config/defaults.yaml            ← project-wide defaults
            └── reads from
                .env                    ← secrets and paths (never committed)
```

All configuration must be loadable without database access so Phase 1 can run standalone.

---

## Dependency Rules

- `ml/` must **not** import from `apps/`, `contribution/`, `unlearning/`, `evaluation/`, or `scheduler/`.
- `contribution/` may import from `ml/` but not from `unlearning/` or `apps/`.
- `unlearning/` may import from `ml/` and `contribution/` but not from `apps/`.
- `apps/api/` may import from all other packages (it is the composition root for the server).
- `apps/fl/` may import from `ml/` and `contribution/` only.
- No circular imports.

---

## Interface Stability Contracts

The following interfaces are defined in Phase 1 and must remain stable across all future phases:

| Interface | Location | Contract |
|-----------|----------|---------|
| `ModelFactory.create(config)` | `ml/models/factory.py` | Returns a `nn.Module` given a config dict |
| `Trainer.train(model, loaders, config)` | `ml/training/trainer.py` | Returns a `TrainingResult` dataclass |
| `Evaluator.evaluate(model, loader)` | `ml/training/evaluator.py` | Returns an `EvaluationResult` dataclass |
| `CheckpointManager.save(model, metadata)` | `ml/training/checkpoint.py` | Saves to disk, returns path and checksum |
| `CheckpointManager.load(path)` | `ml/training/checkpoint.py` | Returns `(model_state_dict, metadata)` |
