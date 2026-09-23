# FedErase — Project Overview

## What is FedErase?

FedErase is a **resource-adaptive federated unlearning framework** for distributed machine learning.

Federated Learning (FL) allows multiple clients to collaboratively train a shared global model without sending their raw data to a central server. Each client trains locally on its own data and sends only model updates (gradients or weights) back to the server, which aggregates them into an improved global model.

### The Problem

After a client participates in federated training, its learned contribution becomes embedded in the global model's weights. If a client later requests deletion of its data (e.g., under GDPR's Right to Erasure), simply deleting the local dataset is insufficient — the model still carries the influence of that client's training.

The naive solution — removing the client and retraining the entire model from scratch — is correct but computationally expensive, communication-heavy, and slow. It grows worse as the number of rounds, clients, and model size increases.

### The FedErase Approach

FedErase aims to:

1. Track **lightweight contribution metadata** for each client during training.
2. **Estimate the influence** of a target client using those metadata.
3. Receive **client-level deletion requests**.
4. Select an **unlearning strategy** based on estimated influence and available resources.
5. Perform **targeted, approximate unlearning** rather than full retraining.
6. Execute **limited repair training** using the remaining clients.
7. **Evaluate** whether the target contribution has been effectively mitigated.
8. **Compare** the result against a full-retraining reference model.
9. **Measure** computation, communication, latency, and storage costs.
10. Support **continuously arriving new local data** and dynamic client events.

### Scientific Position (MANDATORY)

> FedErase does **not** claim to guarantee exact deletion of all knowledge from the model.
>
> FedErase **estimates and mitigates** a target client's learned contribution and evaluates the resulting model against a reference model trained without that client. Unlearning effectiveness is experimentally demonstrated, not mathematically proven.

This position must be reflected in all code comments, documentation, log messages, API responses, and UI text throughout the project.

---

## Project Phases

| Phase | Scope |
|-------|-------|
| 1 | Centralized PyTorch baseline (CIFAR-10) |
| 2 | Flower federated learning (IID) |
| 3 | Non-IID data partitioning |
| 4 | Client contribution tracking |
| 5 | Full-retraining reference baseline |
| 6 | FedErase v1 unlearning |
| 7 | Repair training |
| 8 | Resource-adaptive strategy selection |
| 9 | Real-time new-data adaptation |
| 10 | Model versioning and scheduler |
| 11 | FastAPI control plane |
| 12 | PostgreSQL persistence |
| 13 | React dashboard |
| 14 | Docker deployment |
| 15 | Full integration testing and benchmarking |

**Current phase: Phase 1.**

---

## Technology Stack

| Layer | Technology |
|-------|-----------|
| Core ML | Python 3.11+, PyTorch, TorchVision, NumPy, Pandas |
| Federated Learning | Flower (flwr) — ServerApp / ClientApp / simulation runtime |
| Dataset | CIFAR-10 (primary); FEMNIST/EMNIST (future) |
| Data partitioning | Flower Datasets — IID and Dirichlet non-IID |
| Backend API | FastAPI |
| Database | PostgreSQL (SQLite permitted in Phase 1 only) |
| Frontend | React + Vite + TypeScript |
| Charts | Recharts |
| Testing | pytest (backend), frontend tests TBD |
| Deployment | Docker + Docker Compose |
| Experiment tracking | Optional — MLflow or W&B, not mandatory |

---

## Repository Layout (target)

```
fed-erase/
├── README.md
├── pyproject.toml
├── .env.example
├── docker-compose.yml
├── Makefile
├── apps/
│   ├── fl/            # Flower ServerApp / ClientApp
│   ├── api/           # FastAPI control plane
│   └── frontend/      # React + Vite dashboard
├── ml/
│   ├── models/        # CNN, model factory
│   ├── training/      # trainer, evaluator, checkpoint
│   └── datasets/      # CIFAR-10 loader, partitioning, transforms
├── contribution/      # Contribution tracker, features, influence, storage
├── unlearning/        # Strategies, repair, pipeline, baseline retraining
├── evaluation/        # Accuracy, forgetting, privacy, efficiency, comparison
├── scheduler/         # Event bus, FL scheduler, eligibility policies
├── database/          # ORM models, session, migrations
├── experiments/       # Configs, scripts, results
├── tests/             # All test modules
├── docker/            # Dockerfiles
└── docs/              # Architecture, algorithm, experiments, API docs
```

---

## Key Contacts / Ownership

This is a research prototype. All design decisions that affect the scientific validity of federated unlearning must be:

1. Explicitly documented in `docs/algorithm.md`.
2. Made configurable via experiment configuration files.
3. Never silently hard-coded.
