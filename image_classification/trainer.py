import click
from pathlib import Path
import mlflow.pytorch
from mlflow.tracking import MlflowClient
import pytorch_lightning as pl
from torch import nn
import torch.nn.functional as F

from datasets import MNISTDataModule
from imageclsmodule import ImageClassificationModule

# TODO: AutoAugment
# TODO: MLFlow VS TensorBoard
# TODO: ONNX inference
# TODO: Using premade network, eg. vgg16. Need to adapt start and end of network.


def print_auto_logged_info(r):
    tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [
        f.path for f in MlflowClient().list_artifacts(r.info.run_id, "model")
    ]
    print("run_id: {}".format(r.info.run_id))
    print("artifacts: {}".format(artifacts))
    print("params: {}".format(r.data.params))
    print("metrics: {}".format(r.data.metrics))
    print("tags: {}".format(tags))


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


@click.command()
@click.option(
    "--mlflow", is_flag=True, help="whether to use mlflow autologging"
)
def train(mlflow):

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
    trainer = pl.Trainer(max_epochs=1, callbacks=callbacks)

    # Train the model
    model = ImageClassificationModule(network=Net())
    if mlflow:
        # Auto log all MLflow entities
        with mlflow.start_run() as run:
            trainer.fit(model, data)

        # fetch the auto logged parameters and metrics
        print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))
    else:
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
