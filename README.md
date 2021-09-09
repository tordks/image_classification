# Image classification

This repo contains a thin abstraction around Pytorch Lightning for training
image classification models. When training a deep learning model there is a lot
of boilerplate code. Pytorch Lightning goes a long way to abstract the training
loop, however there still challenges like reproducability and model versioning.

The idea of this training tool is to configure training runs throguh
configuration files and with as little as much boilerplate be able to reuse
functionality. The configuration files can be easily differenced visually and
saved to the trained model.

The implementation is built around the fact that python can do dynamic imports
and instantiation at runtime. Within the configuration file the object to be
instantiated are given. As an example, a LightningModule can be defined as follows:

```
module:
  network:
    name: image_classification.network.MNISTNet
  loss:
    name: torch.nn.CrossEntropyLoss
  optimizer:
    name: torch.optim.Adam
    kwargs:
      lr: 0.01
  lr_scheduler:
    name: torch.optim.lr_scheduler.ReduceLROnPlateau
    monitor: "val_loss"
```

To change the `Adam` optimizer to another one only need to replace the
`optimizer` dictionary. As an example, let us assume we have found an optimizer
implementation we would like to reuse, eg.
[AdaBelief](https://github.com/juntang-zhuang/Adabelief-Optimizer), then the
updated optimizer dictonary would look like:

```
  optimizer:
    name: adabelief_pytorch.AdaBelief
    kwargs:
      lr: 0.01
```

Note that the `adabelief_pytorch` python package needs to be installed in the
environment.

Easy changing/overview of hyperparameters is supported through yaml references:

```
hyperparameters:
    lr &lr: 0.02

module:
    optimizer:
        name: torch.optim.Adam
        kwargs:
            lr: *lr
```

This also opens up for configurable hyperparameter search (not yet implemented).
Given a set of hyperparameters one can generate configuration files and schedule
as many runs as one wants with a single call. The outputs of these runs are
then saved alongside logged metrics and the configuration file used and can
later be visualized and compared with eg. TensorBoard.

**Disclaimer:** The implementation is my own (unless otherwise stated), but the
different ideas and techniques are consolidated from different sources.
