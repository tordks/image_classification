import click
from pathlib import Path
import pytorch_lightning as pl
from torch import nn
import torch.nn.functional as F

from image_classification.datasets import MNISTDataModule
from image_classification.imageclsmodule import ImageClassificationModule

# TODO: AutoAugment
# TODO: MLFlow VS TensorBoard
# TODO: ONNX inference
# TODO: Using premade network, eg. vgg16. Need to adapt start and end of network.


class Net(nn.Module):
    """
    Network from pytorch mnist example
    """

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# TODO: Load Trainer options from config file
@click.command()
@click.option("--max-epochs", type=int, help="max epochs to use for training.")
@click.option(
    "--max-time",
    type=str,
    help="max time to use for training. On the format 'DD.HH.MM.SS'",
)
@click.option("--seed", type=int, help="number to use for setting random seed")
def train(seed, max_epochs, max_time):

    if seed is not None:
        pl.seed_everything(seed, workers=True)
    # Set up data
    data_dir = "~/exploration/image_classification/data"
    data = MNISTDataModule(data_dir)

    # Set up trainer
    callbacks = [
        pl.callbacks.ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            every_n_epochs=1,
        ),
        pl.callbacks.LearningRateMonitor(
            logging_interval="step",
            log_momentum=True,
        ),
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
    model = ImageClassificationModule(network=Net())
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
