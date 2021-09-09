from image_classification.util import dynamic_loader
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
    def __init__(self, config):
        """
        Module for performing image classification.
        """
        # TODO: Define config conventions through eg. pydantic
        super().__init__()
        self.config = config
        self.network = dynamic_loader(self.config["network"])

    def setup(self, stage):
        """
        Sets up the state
        """
        self.loss = dynamic_loader(self.config["loss"])

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
        optimizer = dynamic_loader(
            self.config["optimizer"], extra_args=[self.network.parameters()]
        )
        configuration = {"optimizer": optimizer}

        if "lr_scheduler" in self.config:
            lr_scheduler = dynamic_loader(
                self.config["lr_scheduler"], extra_args=[optimizer]
            )
            configuration["lr_scheduler"] = lr_scheduler

            if "monitor" in self.config["lr_scheduler"]:
                configuration["monitor"] = self.config["lr_scheduler"][
                    "monitor"
                ]

        return configuration
