from pathlib import Path
from typing import Optional

import hydra
import onnx
import pytorch_lightning as pl
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from ruamel.yaml import YAML

from image_classification import logger
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
        model.save_hyperparameters(hyperparameters)

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


def get_hyperparameter(current_config: DictConfig, hydra_config: DictConfig):
    """
    Within the config we don't know which values were the hyperparameters. Hence
    extract the hyperparameter keys from the search space and get the values
    from the config.
    """
    search_space = hydra_config.sweeper.params
    hyperparameters = {
        simplify_search_space_key(hparam): simplify_search_space_value(
            deep_get(current_config, hparam)
        )
        for hparam in search_space
    }
    return hyperparameters


# TODO: How to make optuna and neptune work well togheter?
# TODO: Make sure all metadata are propagated to Neptune
#   * optimization results file (need to happen outside run function)
#   * .hydra folder with hydra overrides and configs
@hydra.main(config_path="./configs/", config_name="config")
def run(config: DictConfig):
    hydra_config = HydraConfig.get()
    logger.info(f"config: {config}")

    hyperparameters = None
    if hydra_config.sweeper.params is not None:
        hyperparameters = get_hyperparameter(config, hydra_config)
        logger.info(f"hyperparameters: {hyperparameters}")

    return train(config, hyperparameters=hyperparameters)


if __name__ == "__main__":
    run()
