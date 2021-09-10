import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from typing import Union
from torch.nn import Module

# TODO: weight loss based on class weights from datamodule
# TODO: weigh loss based on class compatibility
# TODO: adaptively weigh hard samples more than easy samples
# TODO: add augmentations as inputs
# TODO: add visualization of test/validation images

ModuleType = Union[Module, pl.LightningModule]


class ImageClassificationModule(pl.LightningModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.network = hydra.utils.instantiate(self.config.network)

    def setup(self, stage):
        """
        Sets up the state
        """
        # TODO: setup vs __init__
        self.loss = hydra.utils.instantiate(self.config.loss)

    def forward(self, x):
        return self.network(x)

    def training_step(self, batch, batch_idx):
        """
        :param batch: dictionary of batch information from dataloader
        :param batch_idx: current batch idx
        """
        logits = self.network(batch["feature"])
        loss = self.loss(logits, batch["label"])
        self.log("loss", loss, on_step=False, on_epoch=True, logger=True)
        return {"batch_idx": batch_idx, "loss": loss}

    def validation_step(self, batch, batch_idx):
        """
        :param batch: dictionary of batch information from dataloader
        :param batch_idx: current batch idx
        """
        logits = self.network(batch["feature"])
        loss = self.loss(logits, batch["label"])
        self.log("val_loss", loss, on_step=False, on_epoch=True, logger=True)
        return {"batch_idx": batch_idx, "val_loss": loss}

    def test_step(self, batch, batch_idx):
        """
        :param batch: dictionary of batch information from dataloader
        :param batch_idx: current batch idx
        """
        logits = self.network(batch["feature"])
        loss = self.loss(logits, batch["label"])
        self.log("test_loss", loss, on_step=False, on_epoch=True, logger=True)
        return {"batch_idx": batch_idx, "test_loss": loss}

    def configure_optimizers(self):
        """
        Configure optimizer and scheduler
        """
        optimizer = hydra.utils.instantiate(
            self.config.optimizer, self.network.parameters()
        )
        configuration = {"optimizer": optimizer}

        if "lr_scheduler" in self.config:
            lr_scheduler = hydra.utils.instantiate(
                self.config.lr_scheduler, optimizer
            )
            configuration["lr_scheduler"] = lr_scheduler

            if "monitor" in self.config:
                configuration["monitor"] = self.config.monitor

        return configuration
