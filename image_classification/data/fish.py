from pathlib import Path
from typing import Optional
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
from torchvision.transforms import Compose, ToTensor, Resize
from torchvision.datasets import ImageFolder

from image_classification.data.wrapper import DatasetWrapper


class FishDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: Path,
        # download: bool = False,
        batch_size: int = 1,
        num_workers: int = 1,
        resize: int = 64,
        train_val_test_split: tuple[int, int, int] = (0.6, 0.2, 0.2)
        # transforms = None
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.resize = resize
        self.train_val_test_split = train_val_test_split

    def setup(self, stage: Optional[str] = None):

        # NOTE: need to resize to get image as a square. Needed by collate in
        # pytorch when collating samples into batches.
        transform = Compose([ToTensor(), Resize((self.resize, self.resize))])

        dataset = ImageFolder(self.data_dir, transform=transform)
        train_val_test_split = [
            round(split * len(dataset)) for split in self.train_val_test_split
        ]
        # TODO: ensure that this is always the case
        # TODO: ensure class distributions match
        assert sum(train_val_test_split) == len(dataset)

        self.dataset = DatasetWrapper(dataset, {0: "feature", 1: "label"})

        # TODO: ensure that each process gets the same split
        self.train, self.val, self.test = random_split(
            self.dataset, train_val_test_split
        )

    def train_dataloader(self):
        return DataLoader(self.train, self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val, self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test, self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.dataset, self.batch_size)

    # def teardown(self, stage: Optional[str] = None):
    #     ...
