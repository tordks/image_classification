from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
from torch.utils.data.dataset import random_split
from torch.utils.data.dataloader import DataLoader
import torchvision
from torchvision import transforms


class MNIST(torchvision.datasets.MNIST):
    """
    Wrap batch into a dictionary
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
        self, data_dir: Path, batch_size: int = 32, num_workers: int = 6
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
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
        full = MNIST(self.data_dir, train=True, transform=self.transforms)
        self.train, self.val = random_split(full, [55000, 5000])

        self.test = MNIST(self.data_dir, train=False, transform=self.transforms)

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
