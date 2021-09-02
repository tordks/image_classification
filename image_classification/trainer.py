import click
from pathlib import Path
import mlflow.pytorch
from mlflow.tracking import MlflowClient
import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F
import torchvision

from datasets import MNIST
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
@click.option("--output-dir", "-o", type=click.Path(exists=True), default=".")
@click.option(
    "--mlflow", is_flag=True, help="whether to use mlflow autologging"
)
def train(output_dir, mlflow):

    output_dir = Path(output_dir)

    # train dataloader
    train_dataset = MNIST(
        root="./data",
        train=True,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=32, shuffle=False, num_workers=6
    )

    # validation dataloader
    val_dataset = MNIST(
        root="./data",
        train=False,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=64, shuffle=False
    )

    model = ImageClassificationModule(network=Net())

    trainer = pl.Trainer(max_epochs=1)

    # Train the model
    if mlflow:
        # Auto log all MLflow entities
        with mlflow.start_run() as run:
            trainer.fit(model, train_dataloader, val_dataloader)

        # fetch the auto logged parameters and metrics
        print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))
    else:
        trainer.fit(model, train_dataloader, val_dataloader)

    # Save model
    input_sample = train_dataset[0]["feature"]
    input_sample = input_sample.reshape((1, *input_sample.shape))
    model.to_onnx(
        output_dir / "mnist_model.onnx",
        input_sample=input_sample,
        input_names=["feature"],
        output_names=["prediction"],
        dynamic_axes={"feature": {0: "batch_size"}},
        export_params=True,
    )


if __name__ == "__main__":
    train()