# Image classification

This repo contains a thin abstraction around Pytorch Lightning for training
image classification models. When training a deep learning model there is a lot
of boilerplate code. Pytorch-Lightning goes a long way to abstract the training
loop, however there still challenges like reproducibility and model versioning.

The idea of this training tool is to configure training runs through
configuration files and with as little as much boilerplate be able to reuse
functionality. The configuration files can be easily differenced visually and
saved to the trained model. The configuration is handled through
[Hydra](hydra.cc) and hence supports it's tooling for configuration management.
Hydra allows for composable configuration, which is utilized through this
training tool.

The implementation is built around the fact that python can do dynamic imports
and instantiation at runtime. Within the configuration file the object to be
instantiated are given. As an example, a LightningModule can be defined as
follows:

```
module:
  network:
    _target_: image_classification.network.CIFAR10Net
  loss:
    _target_: image_classification.losses.FocalLoss
    gamma: 2
  optimizer:
    _target_: torch.optim.Adam
    lr: 0.01
  lr_scheduler:
    _target_: torch.optim.lr_scheduler.ReduceLROnPlateau
  monitor: "val_loss"
```

To change the `Adam` optimizer to another one only need to replace the
`optimizer` dictionary. As an example, let us assume we have found an optimizer
implementation we would like to reuse, eg.
[AdaBelief](https://github.com/juntang-zhuang/Adabelief-Optimizer), then the
updated optimizer dictonary would look like:

```
  optimizer:
    _target_: adabelief_pytorch.AdaBelief
    lr: 0.01
```

Note that the `adabelief_pytorch` python package needs to be installed in the
environment.


**Disclaimer:** The implementation is my own (unless otherwise stated), but the
different ideas and techniques are consolidated from different sources.
