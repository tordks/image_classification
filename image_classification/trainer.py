import click
from pathlib import Path
import pytorch_lightning as pl
from ruamel.yaml import YAML

from image_classification.util import dynamic_loader
from image_classification.imageclsmodule import ImageClassificationModule

# TODO: AutoAugment
# TODO: MLFlow VS TensorBoard
# TODO: ONNX inference
# TODO: Using premade network, eg. vgg16. Need to adapt start and end of network.


# TODO: Load Trainer options from config file
@click.command()
@click.argument("config")
@click.option("--max-epochs", type=int, help="max epochs to use for training.")
@click.option(
    "--max-time",
    type=str,
    help="max time to use for training. On the format 'DD.HH.MM.SS'",
)
def train(config, max_epochs, max_time):

    # TODO: Wrap config in a predictable object, eg. using pydantic
    config = YAML().load(Path(config).read_text())
    seed = config["framework"]["seed"]

    if seed is not None:
        pl.seed_everything(seed, workers=True)

    # Set up data
    data = dynamic_loader(config["data"])

    # Set up trainer
    callbacks = [
        dynamic_loader(attribute_config)
        for _, attribute_config in config["framework"]["callbacks"].items()
    ]

    tb_logger = pl.loggers.TensorBoardLogger(
        save_dir=Path().resolve(),
        name="lightning_logs",
        default_hp_metric=False,
    )
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        max_time=max_time,
        callbacks=callbacks,
        logger=tb_logger,
        profiler="pytorch",
    )

    # Train the model
    model = ImageClassificationModule(config["training"]["module"])
    trainer.fit(model, data)

    # Save model
    input_sample = data.train[0]["feature"]
    input_sample = input_sample.reshape((1, *input_sample.shape))
    model.to_onnx(
        Path(model.logger.experiment.log_dir) / "mnist_model.onnx",
        input_sample=input_sample,
        input_names=["feature"],
        output_names=["prediction"],
        dynamic_axes={"feature": {0: "batch_size"}},
        export_params=True,
    )


if __name__ == "__main__":
    train()
