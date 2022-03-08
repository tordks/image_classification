from copy import deepcopy
from typing import Optional

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

import onnx
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.utilities.parsing import AttributeDict
from ruamel.yaml import YAML

from image_classification.imageclsmodule import ImageClassificationModule
from image_classification.utils import (
    deep_get,
    simplify_search_space_key,
    simplify_search_space_value,
)


def train(config: DictConfig, hyperparameters: Optional[dict] = None):
    """
    Pipeline for training an image classification model
    """
    # TODO: add check of required keys in config.

    if "seed" in config:
        pl.seed_everything(config["seed"], workers=True)

    training_logger = hydra.utils.instantiate(config.training_logger)

    config_path = Path("train_config.yaml")
    YAML().dump(OmegaConf.to_object(config), config_path)
    training_logger.save_file(key=config_path.name, fpath=config_path)

    # Set up data
    data: pl.LightningDataModule = hydra.utils.instantiate(config.data)

    # Set up training
    callbacks = [
        hydra.utils.instantiate(callback)
        for _, callback in config.callbacks.items()
    ]

    trainer = hydra.utils.instantiate(
        config.trainer,
        logger=training_logger,
        callbacks=callbacks,
    )

    model: pl.LightningModule = ImageClassificationModule(config.module)
    if hyperparameters:
        # Could (or maybe should) set these in the pl Module, but there are
        # hparams that are not inside the module, eg. batch size. So this seems
        # more appropriate
        model._set_hparams(AttributeDict(hyperparameters))
        model._hparams_initial = deepcopy(model._hparams)

    # Train
    trainer.fit(model, data)
    # NOTE: Enabling the test run overwrites the hp_metric from the validation
    # run.
    # trainer.test(model, data)

    input_batch_example = model.prepare_batch(
        next(iter(data.train_dataloader()))
    )
    input_sample = input_batch_example["feature"][0]
    input_sample = input_sample.reshape((1, *input_sample.shape))

    model_path = Path("model.onnx")
    model.to_onnx(
        model_path,
        input_sample=input_sample,
        input_names=["feature"],
        output_names=["prediction"],
        dynamic_axes={"feature": {0: "batch_size"}},
        export_params=True,
    )
    onnx_model = onnx.load(model_path)
    # TODO: check here if batch size is dynamic
    onnx.checker.check_model(onnx_model)

    trainer.logger.save_file(key="model.onnx", fpath=model_path)

    metric_to_optimize = config.get("metric_to_optimize")
    if metric_to_optimize is not None:
        if metric_to_optimize not in trainer.callback_metrics:
            raise ValueError(f"{metric_to_optimize} no available for hp search")
        return trainer.callback_metrics[metric_to_optimize]


# TODO: How to make optuna and neptune work well togheter?
# TODO: Make sure all metadata are propagated to Neptune
#   * optimization results file (need to happen outside run function)
#   * .hydra folder with hydra overrides and configs
@hydra.main(config_path="./configs/", config_name="config")
def run(config: DictConfig):
    hydra_config = HydraConfig.get()
    if "sweeper" in hydra_config and "search_space" in hydra_config.sweeper:
        search_space = hydra_config.sweeper.search_space
        hyperparameters = {
            simplify_search_space_key(hparam): simplify_search_space_value(
                deep_get(config, hparam)
            )
            for hparam in search_space
        }
        train_result = train(config, hyperparameters=hyperparameters)
    else:
        train_result = train(config)

    return train_result


if __name__ == "__main__":
    run()
