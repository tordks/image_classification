from image_classification.metrics import MetricsWrapper
import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl

from typing import Union
from torch.functional import Tensor
from torch.nn import Module
from torchmetrics import Metric

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
        self.training_metrics = {}
        if "training_metrics" in self.config:
            self.training_metrics: dict[str, Metric] = {
                name: hydra.utils.instantiate(metric)
                for name, metric in self.config.training_metrics.items()
            }
            self.setup_metrics(self.training_metrics)

        self.validation_metrics = {}
        if "validation_metrics" in self.config:
            self.validation_metrics: dict[str, Metric] = {
                name: hydra.utils.instantiate(metric)
                for name, metric in self.config.validation_metrics.items()
            }
            self.setup_metrics(self.validation_metrics)

    def setup_metrics(self, metrics):
        for metric_name, metric in metrics.items():
            if not isinstance(metric, MetricsWrapper):
                metric = MetricsWrapper(metric)
                metrics[metric_name] = metric

    def forward(self, x):
        return self.network(x)

    def update_metrics(
        self, metrics: dict[str, Metric], prediction: Tensor, label: Tensor
    ):
        for metric in metrics.values():
            metric.to(prediction.device)
            # NOTE: assume, all metrics only take in prediction/label as args.
            metric.update(prediction, label)

    def log_metrics(self, metrics: list[Metric]):
        for metric_name, metric in metrics.items():
            metric_value = metric.compute()
            if metric_name == self.config.hp_metric:
                # The hp_metric is the default name which is propagated in to
                # the hp dashboard.
                self.log("hp_metric", metric_value, logger=True)
            self.log(metric_name, metric_value, logger=True)

            metric.reset()

    def training_step(self, batch, batch_idx):
        """
        :param batch: dictionary of batch information from dataloader
        :param batch_idx: current batch idx
        """
        prediction = self.network(batch["feature"])
        loss = self.loss(prediction, batch["label"])
        self.log("loss", loss, on_step=False, on_epoch=True, logger=True)
        self.update_metrics(self.training_metrics, prediction, batch["label"])

        return {"batch_idx": batch_idx, "loss": loss}

    def validation_step(self, batch, batch_idx):
        """
        :param batch: dictionary of batch information from dataloader
        :param batch_idx: current batch idx
        """
        prediction = self.network(batch["feature"])
        loss = self.loss(prediction, batch["label"])
        self.log("val_loss", loss, on_step=False, on_epoch=True, logger=True)
        self.update_metrics(self.validation_metrics, prediction, batch["label"])

        return {"batch_idx": batch_idx, "val_loss": loss}

    def on_train_epoch_end(self):
        self.log_metrics(self.training_metrics)

    def on_validation_epoch_end(self):
        self.log_metrics(self.validation_metrics)

    def test_step(self, batch, batch_idx):
        """
        :param batch: dictionary of batch information from dataloader
        :param batch_idx: current batch idx
        """
        prediction = self.network(batch["feature"])
        loss = self.loss(prediction, batch["label"])
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
