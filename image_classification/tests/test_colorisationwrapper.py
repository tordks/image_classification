from pathlib import Path

import numpy as np
import pytest
from torchvision.datasets import CIFAR10

from image_classification.data.colorisation import ColorisationWrapper

DATADIR = Path("data")


@pytest.mark.skipif(
    not (DATADIR / "cifar-10-batches-py").exists(),
    reason="Don't want to download cifar10 in CI",
)
def test_colorisation_wrapper():
    cifar10 = CIFAR10(root=DATADIR, download=False)
    dataset = ColorisationWrapper(cifar10, {0: "feature", 1: "target"})
    sample = dataset[0]

    feature = np.asarray(sample["feature"])
    target = np.asarray(sample["target"])

    assert target.ndim == 3
    assert feature.ndim == 2
    assert target.shape[-1] == 3
    assert target.shape[:-1] == feature.shape
