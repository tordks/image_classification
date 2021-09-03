from loguru import logger
import pytorch_lightning as pl
from ruamel.yaml import YAML
from typing import Union
import torch
from torch.nn import Module
from torch.optim.lr_scheduler import ReduceLROnPlateau


ModuleType = Union[Module, pl.LightningModule]


class ImageClassificationModule(pl.LightningModule):
    def __init__(self, network: ModuleType):
        super().__init__()
        self.network = network

    def setup(self, stage):
        self.loss: torch.nn.CrossEntropyLoss = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        return self.network(x)

    def training_step(self, batch, batch_idx):
        """
        :param batch: dictionary of batch information from dataloader
        :param batch_idx: current batch idx
        """
        logits = self.network(batch["feature"])
        loss = self.loss(logits, batch["label"])

        return {"batch_idx": batch_idx, "loss": loss}

    def validation_step(self, *args, **kwargs):
        # TODO: visualize on validation set every nth batch
        return super().validation_step(*args, **kwargs)

    def training_epoch_end(self, outputs):
        # TODO: visualize metrics every epoch
        ...

    def configure_optimizers(self):
        # TODO: Configure optimizer through class_loader
        optimizer = torch.optim.Adam(self.network.parameters(), 0.001)
        scheduler = ReduceLROnPlateau(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }
