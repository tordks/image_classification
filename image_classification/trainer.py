import hydra
from omegaconf import DictConfig
import onnx
from pathlib import Path
import pytorch_lightning as pl

from image_classification.imageclsmodule import ImageClassificationModule


def train(config: DictConfig):
    """
    Pipeline for trianing an image classification model
    """
    # TODO: add check of required keys in config.

    if "seed" in config:
        pl.seed_everything(config["seed"], workers=True)

    # Set up data
    data: pl.LightningDataModule = hydra.utils.instantiate(
        config.experiment.data
    )

    # Set up training
    callbacks = [
        hydra.utils.instantiate(callback)
        for _, callback in config.callbacks.items()
    ]

    training_logger = hydra.utils.instantiate(config.training_logger)
    log_dir = Path(training_logger.log_dir)

    trainer = hydra.utils.instantiate(
        config.trainer,
        logger=training_logger,
        callbacks=callbacks,
    )

    model: pl.LightningModule = ImageClassificationModule(
        config.experiment.module
    )

    # Train
    trainer.fit(model, data)
    trainer.test(model, data)

    input_sample = data.train[0]["feature"]
    input_sample = input_sample.reshape((1, *input_sample.shape))
    model_path = log_dir / "model.onnx"
    model.to_onnx(
        model_path,
        input_sample=input_sample,
        input_names=["feature"],
        output_names=["prediction"],
        dynamic_axes={"feature": {0: "batch_size"}},
        export_params=True,
    )

    onnx_model = onnx.load(model_path)
    onnx.checker.check_model(onnx_model)


@hydra.main(config_path="./configs/", config_name="config")
def run(config: DictConfig):
    train(config)


if __name__ == "__main__":
    run()
