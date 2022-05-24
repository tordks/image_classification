from typing import Union

import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl

from torch.functional import Tensor
from torch.nn import Module
from torchmetrics import Metric

from image_classification.metrics import MetricsWrapper
from image_classification.utils import prepare_targets
from image_classification.visualization import Stage


# TODO: weight loss based on class weights from datamodule
# TODO: weigh loss based on class compatibility
# TODO: adaptively weigh hard samples more than easy samples
# TODO: add augmentations as inputs
# TODO: consider separating out setup functions to make class more readable.
# TODO: Find a way to get an easy overview of the class state, ie. which self
#       variables exist and the config.

ModuleType = Union[Module, pl.LightningModule]


# TODO: consider refactoring helper functions to make the class more readable
# TODO: How to handle the autograd? (if dont detach: keep the same graph,
# Lightning metrics detach on their own) (no detach => a reference is always
# kept => mem leak)
# TODO: How to handle the different devices? (resource constraint + libs not
# doing GPU)
class ImageClassificationModule(pl.LightningModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.network = hydra.utils.instantiate(self.config.network)

        self.batch_mapping = None
        if "batch_mapping" in self.config:
            if "feature" not in self.config.batch_mapping.values():
                raise ValueError("'feature' must be in the new batch keys")

            if "label" not in self.config.batch_mapping.values():
                raise ValueError("'label' must be in the new batch keys")

            self.batch_mapping = self.config.batch_mapping

        self.visualizations = []
        self.training_metrics = {}
        self.validation_metrics = {}

    # TODO: setup vs __init__. Where is the proper place for the network?
    def setup(self, stage):
        """
        Sets up the Module state
        """
        self.loss = hydra.utils.instantiate(self.config.loss)
        self.setup_metrics()
        self.setup_visualization()

    def setup_metrics(self):
        """
        Instantates all metrics and wraps all metrics in a MetricsWrapper
        """

        def wrap_metrics(metrics):
            for metric_name, metric in metrics.items():
                if not isinstance(metric, MetricsWrapper):
                    metric = MetricsWrapper(metric)
                    metrics[metric_name] = metric

        if "training_metrics" in self.config:
            self.training_metrics: dict[str, Metric] = {
                name: hydra.utils.instantiate(metric)
                for name, metric in self.config.training_metrics.items()
            }
            wrap_metrics(self.training_metrics)

        if "validation_metrics" in self.config:
            self.validation_metrics: dict[str, Metric] = {
                name: hydra.utils.instantiate(metric)
                for name, metric in self.config.validation_metrics.items()
            }
            wrap_metrics(self.validation_metrics)

    def setup_visualization(self):
        """
        Instantiates all visualizations
        """
        if "visualization" in self.config:
            for plotter_config in self.config.visualization:
                plotter = hydra.utils.instantiate(plotter_config)
                self.visualizations.append(plotter)

    def visualize(self, data: dict[str], stage: Stage, step: int):
        # TODO: make into callback?
        """
        Create all visualizations for the specified stage.

        :param data: Data available for us in the visualization
        :param stage: The stage from which this is called
        :param step: The current step. Meaning depends on stage
        """
        # TODO: viz expensive, some processing should only happen every nth,
        # when the viz is needed
        for plotter in self.visualizations:
            if plotter.stage == stage and step % plotter.every_n == 0:
                plot_data = prepare_targets(data, plotter.targets)

                for key, value in plot_data.items():
                    if isinstance(value, Metric):
                        plot_data[key] = value.compute()

                figure = plotter.plot(**plot_data)
                self.logger.log_image(
                    f"{stage.value}/{plotter.identifier}",
                    figure,
                    self.global_step,
                )

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
            if metric.log:
                metric_value = metric.compute()
                if metric_name == self.config.hp_metric:
                    # The hp_metric is the default name which is propagated in
                    # to the hp dashboard.
                    self.log("hp_metric", metric_value, logger=True)
                self.log(metric_name, metric_value, logger=True)

                metric.reset()

    def prepare_batch(self, batch: Union[list, dict]):
        """
        When reusing a datamodule someone else have created, the batch output
        might be on a completely different format than we expect. Map the keys
        from the batch into the required convention.
        """
        if self.batch_mapping is None:
            return batch
        else:
            return {
                new_key: batch[key]
                for key, new_key in self.batch_mapping.items()
            }

    def training_step(self, batch, batch_idx):
        """
        :param batch: dictionary of batch information from dataloader
        :param batch_idx: current batch idx
        """
        batch = self.prepare_batch(batch)

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
        batch = self.prepare_batch(batch)

        prediction = self.network(batch["feature"])
        loss = self.loss(prediction, batch["label"])
        self.log("val_loss", loss, on_step=False, on_epoch=True, logger=True)
        self.update_metrics(self.validation_metrics, prediction, batch["label"])

        return {"batch_idx": batch_idx, "val_loss": loss}

    def on_train_epoch_end(self):
        self.log_metrics(self.training_metrics)

    def on_validation_epoch_end(self):
        self.log_metrics(self.validation_metrics)
        # TODO: consider difference between epoch and iteration (# step)
        # TODO: add option of preprocessing before visualization.
        #     * only process on demand to avoid overhead
        self.visualize(
            data=self.validation_metrics,
            stage=Stage.on_validation_epoch_end,
            step=self.current_epoch,
        )

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

        See https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers  # noqa: 501
        for how the input should be. Here we enforce that the
        lr_scheduler_config option
        """
        optimizer = hydra.utils.instantiate(
            self.config.optimizer, self.network.parameters()
        )
        configuration = {"optimizer": optimizer}

        if "lr_scheduler" in self.config:
            scheduler = hydra.utils.instantiate(
                self.config.lr_scheduler["scheduler"], optimizer
            )
            configuration["lr_scheduler"] = dict(self.config.lr_scheduler) | {
                "scheduler": scheduler
            }

        return configuration
