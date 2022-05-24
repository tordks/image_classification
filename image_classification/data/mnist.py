from pathlib import Path
from typing import Callable, Optional

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
# TODO: make typed object that represent the item in a batch


# TODO: How to handle preproc that should be saved with the network,
# (ie. standardization)? Wrapper that contains a subnetwork + tansforms?

# TODO: with new batch_mapping this wrapper dataset might no longer be needed.
class MNIST(torchvision.datasets.MNIST):
    """
    wrap batch into dictionary and standardize feature/label names
    """

    def __getitem__(self, index: int):
        feature_key = "feature"
        label_key = "label"
        item = {
            key: value
            for key, value in zip(
                (feature_key, label_key), super().__getitem__(index)
            )
        }

        return item


class MNISTDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: Path,
        download: bool = False,
        transform: Optional[list[Callable]] = None,
        target_transform: Optional[list[Callable]] = None,
        batch_size: int = 32,
        num_workers: int = 1,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.download = download

        self.transform = transform
        if transform is not None:
            self.transform = transforms.Compose(transform)
        else:
            transform = transforms.ToTensor()

        self.target_transform = target_transform
        if target_transform is not None:
            self.target_transform = transforms.Compose(target_transform)

    def setup(self, stage: Optional[str] = None):
        full = MNIST(
            self.data_dir,
            train=True,
            transform=self.transform,
            target_transform=self.target_transform,
            download=self.download,
        )
        n_samples = int(len(full))
        n_train_samples = int(0.8 * n_samples)
        n_val_samples = n_samples - n_train_samples
        self.train, self.val = random_split(
            full, [n_train_samples, n_val_samples]
        )
        self.test = MNIST(
            self.data_dir,
            train=False,
            transform=self.transform,
            target_transform=self.target_transform,
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
