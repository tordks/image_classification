from typing import Union

import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl

from torch.functional import Tensor
from torch.nn import Module
from torchmetrics import Metric

from image_classification import Stage
from image_classification.metrics import MetricsWrapper
from image_classification.transforms import TransformScheduler
from image_classification.visualization import VisualizationScheduler


# TODO: weight loss based on class weights from datamodule
# TODO: weigh loss based on class compatibility
# TODO: adaptively weigh hard samples more than easy samples
# TODO: consider separating out setup functions to make class more readable.
# TODO: Find a way to get an easy overview of the class state, ie. which self
#       variables exist and the config.


class ImageClassificationModule(pl.LightningModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.network: Module = hydra.utils.instantiate(self.config.network)

        self.batch_mapping = None
        if "batch_mapping" in self.config:
            if "feature" not in self.config.batch_mapping.values():
                raise ValueError("'feature' must be in the new batch keys")

            if "target" not in self.config.batch_mapping.values():
                raise ValueError("'target' must be in the new batch keys")

            self.batch_mapping = self.config.batch_mapping

        self.transform_scheduler = None
        self.visualization_scheduler = None
        self.training_metrics = {}
        self.validation_metrics = {}

    # TODO: setup vs __init__. Where is the proper place for the network?
    def setup(self, stage):
        """
        Sets up the Module state
        """
        self.loss = hydra.utils.instantiate(self.config.loss)
        self.setup_metrics()
        self.setup_transforms()
        self.setup_visualization()

    def setup_metrics(self):
        # TODO: introduce MetricsCollection for handling metrics
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
        Instantiates visualizations
        """
        visualizations = []
        if "visualization" in self.config:
            for plotter_config in self.config.visualization:
                plotter = hydra.utils.instantiate(
                    plotter_config, _convert_="all"
                )
                visualizations.append(plotter)

        self.visualization_scheduler = VisualizationScheduler(visualizations)

    def setup_transforms(self):
        """
        Instantiates transforms
        """
        transforms = []
        if "transforms" in self.config:
            for transform_config in self.config.transforms:
                transform = hydra.utils.instantiate(transform_config)
                transforms.append(transform)
        self.transform_scheduler = TransformScheduler(transforms)

    def transform(self, data: dict[str], stage: Stage, step: int):
        """
        Perform processing at specified stage and step
        """
        return self.transform_scheduler(data, stage, step)

    def visualize(self, data: dict[str], stage: Stage, step: int):
        """
        Create all visualizations for the specified stage and step.

        :param data: Data available for us in the visualization
        :param stage: The stage from which this is called
        :param step: The current step. Meaning depends on stage
        """
        figures = self.visualization_scheduler(data, stage, step)
        for identifier, figure in figures.items():
            self.logger.log_image(
                f"{stage.value}/{identifier}",
                figure,
                self.global_step,
            )

    def forward(self, x):
        return self.network(x)

    def update_metrics(
        self, metrics: dict[str, Metric], prediction: Tensor, target: Tensor
    ):
        for metric in metrics.values():
            metric.to(prediction.device)
            # NOTE: assume, all metrics only take in prediction/target as args.
            metric.update(prediction, target)

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
        batch = batch | self.transform(
            data=batch,
            stage=Stage.training_step_before_predict,
            step=self.global_step,
        )

        prediction = self.network(batch["feature"])
        loss = self.loss(prediction, batch["target"])
        self.log("loss", loss, on_step=False, on_epoch=True, logger=True)
        self.update_metrics(self.training_metrics, prediction, batch["target"])

        data = batch | {"prediction": prediction}
        data = data | self.transform(
            data=data,
            stage=Stage.training_step_after_predict,
            step=self.global_step,
        )

        self.visualize(
            data=data,
            stage=Stage.training_step_after_predict,
            step=self.global_step,
        )
        return {"batch_idx": batch_idx, "loss": loss}

    def validation_step(self, batch, batch_idx):
        """
        :param batch: dictionary of batch information from dataloader
        :param batch_idx: current batch idx
        """
        batch = self.prepare_batch(batch)
        batch = batch | self.transform(
            data=batch,
            stage=Stage.validation_step_before_predict,
            step=self.global_step,
        )

        prediction = self.network(batch["feature"])
        loss = self.loss(prediction, batch["target"])
        self.log("val_loss", loss, on_step=False, on_epoch=True, logger=True)
        self.update_metrics(
            self.validation_metrics, prediction, batch["target"]
        )

        data = batch | {"prediction": prediction}
        data = data | self.transform(
            data=data,
            stage=Stage.validation_step_after_predict,
            step=self.global_step,
        )

        self.visualize(
            data=data,
            stage=Stage.validation_step_after_predict,
            step=self.global_step,
        )

        return {"batch_idx": batch_idx, "val_loss": loss}

    def on_train_epoch_end(self):
        self.log_metrics(self.training_metrics)
        self.visualize(
            data=self.training_metrics,
            stage=Stage.on_train_epoch_end,
            step=self.current_epoch,
        )

    def on_validation_epoch_end(self):
        self.log_metrics(self.validation_metrics)
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
        loss = self.loss(prediction, batch["target"])
        self.log("test_loss", loss, on_step=False, on_epoch=True, logger=True)

        data = batch | {"prediction": prediction}
        data = data | self.transform(
            data=data,
            stage=Stage.validation_step,
            step=self.global_step,
        )

        self.visualize(
            data=data,
            stage=Stage.validation_step,
            step=self.global_step,
        )
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
