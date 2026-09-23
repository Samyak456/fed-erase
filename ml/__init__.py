"""FedErase ML core package.

This package contains the machine learning components shared across all phases:
- models: CNN architecture and model factory
- datasets: CIFAR-10 loader, partitioning, and transforms
- training: Trainer, Evaluator, and CheckpointManager

This package has NO imports from apps/, contribution/, unlearning/,
evaluation/, or scheduler/. It must remain independently usable.
"""
