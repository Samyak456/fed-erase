"""Data transforms for CIFAR-10.

Provides composition of torchvision transforms for training and evaluation pipelines.
Training transforms include optional augmentation (random crop, horizontal flip).
Evaluation transforms apply normalization only.

All transform parameters (normalisation constants, augmentation settings) are
research-critical and documented here.
"""

from __future__ import annotations

from torchvision import transforms

# ---------------------------------------------------------------------------
# CIFAR-10 Normalisation Constants
# ---------------------------------------------------------------------------

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
"""Mean pixel values for CIFAR-10 training set (RGB channels)."""

CIFAR10_STD = (0.2470, 0.2435, 0.2616)
"""Standard deviation of pixel values for CIFAR-10 training set (RGB channels)."""


# ---------------------------------------------------------------------------
# Transform Factories
# ---------------------------------------------------------------------------


def get_train_transforms(augment: bool = True) -> transforms.Compose:
    """Build training transforms for CIFAR-10.

    If `augment=True`, applies random crop (with padding) and horizontal flip
    before converting to tensor and normalising. If `augment=False`, applies
    only tensor conversion and normalisation.

    Args:
        augment: Whether to apply data augmentation (random crop + flip).

    Returns:
        torchvision.transforms.Compose object for the training pipeline.

    Notes:
        - Augmentation: RandomCrop(32, padding=4) fills with 0-padding around edges,
          then crops a random 32x32 region, creating slight jitter.
        - Augmentation: RandomHorizontalFlip(p=0.5) flips with 50% probability.
        - Normalisation is applied identically regardless of augment setting.

    """
    if augment:
        return transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
            ]
        )


def get_eval_transforms() -> transforms.Compose:
    """Build evaluation (validation/test) transforms for CIFAR-10.

    Applies only tensor conversion and normalisation — no augmentation.
    Used for validation and test datasets to ensure deterministic evaluation.

    Returns:
        torchvision.transforms.Compose object for the evaluation pipeline.

    Notes:
        - No augmentation is applied in evaluation mode.
        - Normalisation uses the same constants as training.
        - Deterministic: same input always produces same output.

    """
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
        ]
    )
