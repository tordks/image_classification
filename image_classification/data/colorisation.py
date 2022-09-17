from pathlib import Path
from typing import Callable, Optional, Union

import PIL
import pytorch_lightning as pl
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import random_split
from torchvision import transforms
from torchvision.datasets import CIFAR10

# from torchvision.transforms import ToTensor
from torchvision.transforms.functional import to_grayscale


class ColorisationWrapper(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        sample_mapping: dict[Union[int, str], str],
        transform: Optional[Callable] = lambda x: x,
        target_transform: Optional[Callable] = lambda x: x,
    ):
        """
        Wrap an image dataset so that it outputs the image as target and a
        grayscale as features. Also wrap samples in a dictionary instead of a
        tuple/list to adhere to the convention of "feature" and "target" keys.

        :param dataset: dataset to wrap.
        :param sample_mapping: mapping from each idx in the dataset sample to a
                               key
        """
        self.dataset = dataset
        self.sample_mapping = sample_mapping
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, sample_idx: int):
        sample = {
            self.sample_mapping[i]: element
            for i, element in enumerate(self.dataset[sample_idx])
        }
        if not isinstance(sample["feature"], PIL.Image.Image):
            raise NotImplementedError(
                "ColorisationWrapper only accepts datasets which outputs"
                f"PIL.IMAGE, not {type(sample['features'])}"
            )

        sample["target"] = sample["feature"]
        sample["target"] = self.target_transform(sample["target"])

        sample["feature"] = to_grayscale(sample["feature"])
        sample["feature"] = self.transform(sample["feature"])

        return sample

    def __len__(self):
        return len(self.dataset)


class ColorisationCIFAR10DataModule(pl.LightningDataModule):
    """
    Datamodule for colorisation problems using CIFAR10.
    """

    def __init__(
        self,
        data_dir: Path,
        download: bool = False,
        train_transform: Optional[list[Callable]] = None,
        train_target_transform: Optional[list[Callable]] = None,
        val_transform: Optional[list[Callable]] = None,
        val_target_transform: Optional[list[Callable]] = None,
        batch_size: int = 32,
        num_workers: int = 1,
        shuffle: bool = True,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.download = download
        self.shuffle = shuffle

        # TODO: consider whether we should always return PIL or Tensor? If
        #       Tensor, then initiate transforms with [ToTensor()].
        self.train_transform = train_transform
        if train_transform is not None:
            self.train_transform = transforms.Compose(train_transform)

        self.train_target_transform = train_target_transform
        if train_target_transform is not None:
            self.train_target_transform = transforms.Compose(
                train_target_transform
            )

        self.val_transform = val_transform
        if val_transform is not None:
            self.val_transform = transforms.Compose(val_transform)

        self.val_target_transform = val_target_transform
        if val_target_transform is not None:
            self.val_target_transform = transforms.Compose(val_target_transform)

    def setup(self, stage: Optional[str] = None):
        cifar10_sample_mapping = {0: "feature", 1: "target"}
        full = ColorisationWrapper(
            CIFAR10(
                self.data_dir,
                train=True,
                download=self.download,
            ),
            sample_mapping=cifar10_sample_mapping,
            transform=self.train_transform,
            target_transform=self.train_target_transform,
        )
        n_samples = int(len(full))
        n_train_samples = int(0.8 * n_samples)
        n_val_samples = n_samples - n_train_samples

        self.train, self.val = random_split(
            full, [n_train_samples, n_val_samples]
        )

        self.test = ColorisationWrapper(
            CIFAR10(
                self.data_dir,
                train=False,
                download=self.download,
            ),
            sample_mapping=cifar10_sample_mapping,
            transform=self.transform,
            target_transform=self.target_transform,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test, batch_size=self.batch_size, num_workers=self.num_workers
        )
