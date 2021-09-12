# Image classification

This repo contains a thin abstraction around Pytorch Lightning for training
image classification models. When training a deep learning model there is a lot
of boilerplate code. Pytorch-Lightning goes a long way to abstract the training
loop, however there still challenges like reproducibility, configurability and
model versioning.

The idea of this training tool is to configure training runs through
configuration files and with as little as much boilerplate be able to reuse
functionality. The configuration files can be easily differenced visually and
saved to the trained model. The configuration is handled through
[Hydra](hydra.cc) and hence supports it's tooling for configuration management.
Hydra allows for composable configuration, which is utilized through this
training tool.

To run the trainer with the default config do:

```
python trainer
```

## Goal of this repository

The goal of this repository is to make an importable trainer which can be
applied to any image classification problem. This alleviates the need to have a
trainer for each experiment, and hence each experiment is only required to
implement the experiment specific parts (network, datasets etc.). If more core
functionality (ie. semi-supervised training schemes) is needed within a project,
it can be added and parametrized in this tool.

There is however some way to go before the goal is reached. In it's current
status, this repository is just an example project with some selected
experiments.


## Changing the config through the command line

The training run can be modified using the Hydra API for changing the config on the commandline. This is useful when switching between experiments, hp search modes or for specifying a short run for testing. Below are some examples:

Run cifar10 experiment:

```
python trainer.py experiment=cifar10
```

Run mnist experiment, but only run 1 epoch:

```
python trainer.py experiment=mnist trainer.max_epochs=1
```

For more information on how this works see the [Hydra](hydra.cc) documentation

## Instantiation of classes

The implementation is built around the fact that python can do dynamic imports
and instantiation at runtime. Within the configuration file the object to be
instantiated are indicated by the `_target_` keyword. As an example, a
LightningModule can be defined as follows:

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

## Hyperparameter search

Hyperparameter search is supported through Hydra's optuna plugin, [Optuna
Sweeper](https://hydra.cc/docs/plugins/optuna_sweeper/). To perform a search
using the default search space on the mnist example run the following:

```
python trainer hp_search=mnist_optuna --multirun
```

Please see the Optune Sweeper documentation on how to define the search spaces.

The result of the hyperparameter search will be saved and can be visualized in tensorboard.


## Disclaimer

The implementation is my own (unless otherwise stated), but the different ideas
and techniques are consolidated from different sources.

After moving to Hydra, inspiration have especially been taken from
https://github.com/ashleve/lightning-hydra-template
