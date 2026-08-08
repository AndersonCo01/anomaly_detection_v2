"""
Step 4 Style Checks - training setup recommendations (advisory, non-blocking).

These tests check whether your training setup follows the recommended
approach from the hints. They will NOT block your CI from passing -
they are informational only.

Recommended setup:
    - nn.CrossEntropyLoss as the loss function
    - optim.Adam as the optimizer
"""
import torch.nn as nn
import torch.optim as optim

from steel_defect.train import setup_training

import pytest


class TinyModel(nn.Module):
    """Minimal model for testing."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 16 * 16, 5),
        )

    def forward(self, x):
        return self.net(x)


@pytest.fixture
def tiny_model():
    return TinyModel()


class TestTrainStyle:
    """Advisory checks - recommended but not required."""

    def test_uses_cross_entropy_loss(self, tiny_model):
        criterion, _ = setup_training(tiny_model)
        assert isinstance(criterion, nn.CrossEntropyLoss), (
            f"Hint: CrossEntropyLoss is recommended for multi-class "
            f"classification, you used {type(criterion).__name__}"
        )

    def test_uses_adam_optimizer(self, tiny_model):
        _, optimizer = setup_training(tiny_model)
        assert isinstance(optimizer, optim.Adam), (
            f"Hint: Adam is a good default optimizer, "
            f"you used {type(optimizer).__name__}"
        )
