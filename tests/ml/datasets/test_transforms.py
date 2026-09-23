"""Tests data transform composition, shape/dtype verification, augmentation behavior,
and normalisation correctness.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision import transforms

from ml.datasets.transforms import (
    CIFAR10_MEAN,
    CIFAR10_STD,
    get_eval_transforms,
    get_train_transforms,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def random_pil_image() -> Image.Image:
    """Create a random PIL Image of shape (32, 32, 3) with uint8 values."""
    img_array = np.random.randint(0, 256, size=(32, 32, 3), dtype=np.uint8)
    return Image.fromarray(img_array)


@pytest.fixture
def white_pil_image() -> Image.Image:
    """Create a white PIL Image (all pixels 255)."""
    img_array = np.full((32, 32, 3), 255, dtype=np.uint8)
    return Image.fromarray(img_array)


@pytest.fixture
def black_pil_image() -> Image.Image:
    """Create a black PIL Image (all pixels 0)."""
    img_array = np.zeros((32, 32, 3), dtype=np.uint8)
    return Image.fromarray(img_array)


# =============================================================================
# Tests for Constants
# =============================================================================


@pytest.mark.unit
class TestConstants:
    """Tests for normalisation constants."""

    def test_mean_is_tuple_of_three(self) -> None:
        """Test that CIFAR10_MEAN is a 3-tuple."""
        assert isinstance(CIFAR10_MEAN, tuple)
        assert len(CIFAR10_MEAN) == 3

    def test_std_is_tuple_of_three(self) -> None:
        """Test that CIFAR10_STD is a 3-tuple."""
        assert isinstance(CIFAR10_STD, tuple)
        assert len(CIFAR10_STD) == 3

    def test_mean_values_are_in_range(self) -> None:
        """Test that mean values are reasonable (between 0 and 1)."""
        for m in CIFAR10_MEAN:
            assert 0.0 <= m <= 1.0

    def test_std_values_are_positive(self) -> None:
        """Test that std values are positive."""
        for s in CIFAR10_STD:
            assert s > 0.0


# =============================================================================
# Tests for get_train_transforms
# =============================================================================


@pytest.mark.unit
class TestGetTrainTransforms:
    """Tests for training transform composition."""

    def test_get_train_transforms_with_augment_true_returns_compose(self) -> None:
        """Test that augment=True returns a Compose object."""
        t = get_train_transforms(augment=True)
        assert isinstance(t, transforms.Compose)

    def test_get_train_transforms_with_augment_false_returns_compose(self) -> None:
        """Test that augment=False returns a Compose object."""
        t = get_train_transforms(augment=False)
        assert isinstance(t, transforms.Compose)

    def test_train_transforms_augment_true_has_4_transforms(self) -> None:
        """Test that augment=True has 4 transforms (crop, flip, tensor, normalize)."""
        t = get_train_transforms(augment=True)
        assert len(t.transforms) == 4

    def test_train_transforms_augment_false_has_2_transforms(self) -> None:
        """Test that augment=False has 2 transforms (tensor, normalize)."""
        t = get_train_transforms(augment=False)
        assert len(t.transforms) == 2

    def test_train_transforms_augment_true_output_shape(
        self, random_pil_image: Image.Image
    ) -> None:
        """Test that augment=True produces correct output shape."""
        t = get_train_transforms(augment=True)
        output = t(random_pil_image)
        assert output.shape == (3, 32, 32)

    def test_train_transforms_augment_false_output_shape(
        self, random_pil_image: Image.Image
    ) -> None:
        """Test that augment=False produces correct output shape."""
        t = get_train_transforms(augment=False)
        output = t(random_pil_image)
        assert output.shape == (3, 32, 32)

    def test_train_transforms_output_dtype(self, random_pil_image: Image.Image) -> None:
        """Test that output is a float32 tensor."""
        t = get_train_transforms(augment=False)
        output = t(random_pil_image)
        assert output.dtype == torch.float32

    def test_train_transforms_augment_true_includes_random_crop(
        self, random_pil_image: Image.Image
    ) -> None:
        """Test that augment=True includes RandomCrop transform."""
        t = get_train_transforms(augment=True)
        # Check that first transform is RandomCrop
        assert isinstance(t.transforms[0], transforms.RandomCrop)

    def test_train_transforms_augment_true_includes_horizontal_flip(
        self, random_pil_image: Image.Image
    ) -> None:
        """Test that augment=True includes RandomHorizontalFlip transform."""
        t = get_train_transforms(augment=True)
        # Check that second transform is RandomHorizontalFlip
        assert isinstance(t.transforms[1], transforms.RandomHorizontalFlip)

    def test_train_transforms_augment_false_does_not_include_augmentation(
        self, random_pil_image: Image.Image
    ) -> None:
        """Test that augment=False does not include RandomCrop or RandomHorizontalFlip."""
        t = get_train_transforms(augment=False)
        # Should only have ToTensor and Normalize
        for transform in t.transforms:
            assert not isinstance(transform, transforms.RandomCrop)
            assert not isinstance(transform, transforms.RandomHorizontalFlip)


# =============================================================================
# Tests for get_eval_transforms
# =============================================================================


@pytest.mark.unit
class TestGetEvalTransforms:
    """Tests for evaluation transform composition."""

    def test_get_eval_transforms_returns_compose(self) -> None:
        """Test that get_eval_transforms returns a Compose object."""
        t = get_eval_transforms()
        assert isinstance(t, transforms.Compose)

    def test_eval_transforms_has_2_transforms(self) -> None:
        """Test that eval transforms has 2 transforms (tensor, normalize)."""
        t = get_eval_transforms()
        assert len(t.transforms) == 2

    def test_eval_transforms_output_shape(self, random_pil_image: Image.Image) -> None:
        """Test that output shape is correct."""
        t = get_eval_transforms()
        output = t(random_pil_image)
        assert output.shape == (3, 32, 32)

    def test_eval_transforms_output_dtype(self, random_pil_image: Image.Image) -> None:
        """Test that output is a float32 tensor."""
        t = get_eval_transforms()
        output = t(random_pil_image)
        assert output.dtype == torch.float32

    def test_eval_transforms_does_not_include_augmentation(
        self, random_pil_image: Image.Image
    ) -> None:
        """Test that eval transforms does not include any augmentation."""
        t = get_eval_transforms()
        for transform in t.transforms:
            assert not isinstance(transform, transforms.RandomCrop)
            assert not isinstance(transform, transforms.RandomHorizontalFlip)

    def test_eval_transforms_is_deterministic(self, random_pil_image: Image.Image) -> None:
        """Test that applying eval transform twice produces identical results."""
        t = get_eval_transforms()
        output1 = t(random_pil_image)
        output2 = t(random_pil_image)
        assert torch.allclose(output1, output2)


# =============================================================================
# Tests for Normalisation Correctness
# =============================================================================


@pytest.mark.unit
class TestNormalisation:
    """Tests for normalisation correctness."""

    def test_white_image_normalised_correctly(self, white_pil_image: Image.Image) -> None:
        """Test that white image (255,255,255) normalises correctly.

        Expected: pixel_norm = (1.0 - mean) / std
        """
        t = get_eval_transforms()
        output = t(white_pil_image)

        # White pixels: 255 / 255 = 1.0 in tensor
        # After normalise: (1.0 - mean) / std
        expected = torch.tensor(
            [
                (1.0 - CIFAR10_MEAN[0]) / CIFAR10_STD[0],
                (1.0 - CIFAR10_MEAN[1]) / CIFAR10_STD[1],
                (1.0 - CIFAR10_MEAN[2]) / CIFAR10_STD[2],
            ]
        )
        expected = expected.unsqueeze(1).unsqueeze(2).expand(3, 32, 32)

        assert torch.allclose(output, expected, atol=1e-5)

    def test_black_image_normalised_correctly(self, black_pil_image: Image.Image) -> None:
        """Test that black image (0,0,0) normalises correctly.

        Expected: pixel_norm = (0.0 - mean) / std = -mean / std
        """
        t = get_eval_transforms()
        output = t(black_pil_image)

        # Black pixels: 0 / 255 = 0.0 in tensor
        # After normalise: (0.0 - mean) / std = -mean / std
        expected = torch.tensor(
            [
                (0.0 - CIFAR10_MEAN[0]) / CIFAR10_STD[0],
                (0.0 - CIFAR10_MEAN[1]) / CIFAR10_STD[1],
                (0.0 - CIFAR10_MEAN[2]) / CIFAR10_STD[2],
            ]
        )
        expected = expected.unsqueeze(1).unsqueeze(2).expand(3, 32, 32)

        assert torch.allclose(output, expected, atol=1e-5)

    def test_normalised_values_in_reasonable_range(self, random_pil_image: Image.Image) -> None:
        """Test that normalised values are in a reasonable range.

        For CIFAR-10, after normalisation, values should roughly be in [-3, 3].
        """
        t = get_eval_transforms()
        output = t(random_pil_image)

        # Allow some margin for extreme values
        assert output.min() >= -4.0
        assert output.max() <= 4.0


# =============================================================================
# Tests for Separation of Train and Eval
# =============================================================================


@pytest.mark.unit
class TestTrainEvalSeparation:
    """Tests to ensure training augmentation is NOT applied to eval."""

    def test_train_augment_true_differs_from_eval(self, random_pil_image: Image.Image) -> None:
        """Test that with probabilistic augmentation, train and eval usually differ.

        Note: This is probabilistic — RandomHorizontalFlip might not flip a given image,
        so we only check that the transforms are different structures.
        """
        train_t = get_train_transforms(augment=True)
        eval_t = get_eval_transforms()

        # Train should have more transforms
        assert len(train_t.transforms) > len(eval_t.transforms)

    def test_train_augment_false_same_as_eval(self, random_pil_image: Image.Image) -> None:
        """Test that train with augment=False and eval produce identical outputs."""
        train_t = get_train_transforms(augment=False)
        eval_t = get_eval_transforms()

        train_output = train_t(random_pil_image)
        eval_output = eval_t(random_pil_image)

        assert torch.allclose(train_output, eval_output)

    def test_eval_not_affected_by_config_augment_param(self, random_pil_image: Image.Image) -> None:
        """Test that get_eval_transforms() is independent of any config.

        Eval transforms should never include augmentation regardless of how
        they're called or what training config says.
        """
        eval_t = get_eval_transforms()
        # Verify it does not include RandomCrop or RandomHorizontalFlip
        for transform in eval_t.transforms:
            assert not isinstance(transform, transforms.RandomCrop)
            assert not isinstance(transform, transforms.RandomHorizontalFlip)
