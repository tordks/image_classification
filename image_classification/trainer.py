import click
import onnx
from pathlib import Path
import pytorch_lightning as pl
from ruamel.yaml import YAML

from image_classification.util import dynamic_loader
from image_classification.imageclsmodule import ImageClassificationModule


@click.command()
@click.argument("config")
def train(config):

    # TODO: Wrap config in a predictable, typed object, eg. using pydantic
    config = YAML().load(Path(config).read_text())
    seed = config["seed"]

    if seed is not None:
        pl.seed_everything(seed, workers=True)

    # Set up data
    data = dynamic_loader(config["data"])

    # Set up training
    callbacks = [
        dynamic_loader(attribute_config)
        for _, attribute_config in config["callbacks"].items()
    ]

    training_logger = dynamic_loader(config["training_logger"])
    log_dir = Path(training_logger.log_dir)

    trainer = dynamic_loader(
        config["trainer"],
        extra_kwargs={"logger": training_logger, "callbacks": callbacks},
    )

    model = ImageClassificationModule(config["module"])

    # Train

    #  log_dir is automatically created during fit, but want to save
    #  config before fit is called.
    log_dir.mkdir()
    with open(log_dir / "config.yaml", "w") as fp:
        YAML().dump(config, fp)

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


if __name__ == "__main__":
    train()
