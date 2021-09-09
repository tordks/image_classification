from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
from torch.utils.data.dataset import random_split
from torch.utils.data.dataloader import DataLoader
import torchvision
from torchvision import transforms

# TODO: add class weights to batch for handling class balance
# TODO: make an abstraction for the dataset wrapper
#    * one dataset that wraps imported datasets.
# TODO: make an abstraction for the datamodule
#    * one datamodule which supports multiple
#    * add train transformation as an input
# TODO: make typed object that represent the batch


class CIFAR10(torchvision.datasets.CIFAR10):
    """
    wrap batch into dictionary and standardize feature/label names
    """

    def __getitem__(self, index: int):
        feature_key = "feature"
        label_key = "label"
        batch = {
            key: value
            for key, value in zip(
                (feature_key, label_key), super().__getitem__(index)
            )
        }
        return batch


class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: Path,
        batch_size: int = 32,
        num_workers: int = 6,
        download: bool = False,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.download = download

        self.transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

    def setup(self, stage: Optional[str] = None):
        full = CIFAR10(
            self.data_dir,
            train=True,
            transform=self.transforms,
            download=self.download,
        )
        self.train, self.val = random_split(
            full, [len(full) * 0.8, len(full) * 0.2]
        )
        self.test = MNIST(
            self.data_dir,
            train=False,
            transform=self.transforms,
            download=self.download,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test, batch_size=self.batch_size, num_workers=self.num_workers
        )


class MNIST(torchvision.datasets.MNIST):
    """
    wrap batch into dictionary and standardize feature/label names
    """

    def __getitem__(self, index: int):
        feature_key = "feature"
        label_key = "label"
        batch = {
            key: value
            for key, value in zip(
                (feature_key, label_key), super().__getitem__(index)
            )
        }
        return batch


class MNISTDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: Path,
        batch_size: int = 32,
        num_workers: int = 6,
        download: bool = False,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.download = download

        self.transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.1307,),  # mean of entire set
                    (0.3081,),  # std of entire set
                ),
            ]
        )

    def setup(self, stage: Optional[str] = None):
        full = MNIST(
            self.data_dir,
            train=True,
            transform=self.transforms,
            download=self.download,
        )
        self.train, self.val = random_split(
            full, [len(full) * 0.8, len(full) * 0.2]
        )
        self.test = MNIST(
            self.data_dir,
            train=False,
            transform=self.transforms,
            download=self.download,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test, batch_size=self.batch_size, num_workers=self.num_workers
        )
