from pathlib import Path

import pytest
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor

from image_classification.data.colorisation import (
    ColorisationCIFAR10DataModule,
    ColorisationWrapper,
)

DATADIR = Path("data")


@pytest.mark.skipif(
    not (DATADIR / "cifar-10-batches-py").exists(),
    reason="Don't want to download cifar10 in CI",
)
def test_colorisation_wrapper():
    cifar10 = CIFAR10(root=DATADIR, download=False)
    dataset = ColorisationWrapper(
        cifar10,
        transform=ToTensor(),
        target_transform=ToTensor(),
        sample_mapping={0: "feature", 1: "target"},
    )
    sample = dataset[0]

    feature = sample["feature"]
    target = sample["target"]

    assert feature.shape[0] == 1
    assert target.shape[0] == 3
    assert target.shape[1:] == feature.shape[1:]


@pytest.mark.skipif(
    not (DATADIR / "cifar-10-batches-py").exists(),
    reason="Only meant as visual check of the colorisation dataset",
)
def test_visualize_cifar10_colorisation():
    cifar10 = CIFAR10(root=DATADIR, download=False)
    dataset = ColorisationWrapper(
        cifar10,
        sample_mapping={0: "feature", 1: "target"},
    )

    testdir = Path("tests") / "out"
    testdir.mkdir(exist_ok=True)
    for idx in list(range(10)):
        sample = dataset[idx]
        with open(testdir / f"{idx}_feature.png", "wb") as fp:
            sample["feature"].save(fp)
        with open(testdir / f"{idx}_target.png", "wb") as fp:
            sample["target"].save(fp)


@pytest.mark.skipif(
    not (DATADIR / "cifar-10-batches-py").exists(),
    reason="Don't want to download cifar10 in CI",
)
def test_cifar10_colorisation_datamodule():
    transform = Compose([ToTensor()])

    ColorisationCIFAR10DataModule(
        data_dir=".",
        download=False,
        train_transform=transform,
        train_target_transform=transform,
        val_transform=transform,
        val_target_transform=transform,
    )
