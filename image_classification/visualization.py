from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional, Union

from fastcore.meta import delegates
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay
from torch import Tensor


# TODO: New name? might be confusing with the LightningModule setup argument
#       being called stage
class Stage(Enum):
    training_step = "training_step"
    validation_step = "validation_step"
    on_train_epoch_end = "on_train_epoch_end"
    on_validation_epoch_end = "on_validation_epoch_end"


@dataclass
class Figure:
    """
    A figure with an associated plotter. Holds information on which arguments
    goes into the plotter and at which stage in the training pipeline a plot
    should be made.

    :param plotter: Callable which plots to a matplotlib figure by plotting to
                    the current figure or through an input axes
    :param stage: The stage at which to perform the plotting
    :param targets: Mapping from the data key within the training pipeline to
                    the name of the kwarg in the plotter.
    :param figure_kwargs: kwargs which goes into the matplotlib.pyplot.figure
    :param every_n: Plot every nth step. The meaning step changes depending on
                    which stage the plot happens for. if stage =
                    "on_train_epoch_end", then step is interpretted as epoch
                    number
    :param input_ax: Wheter to input an "ax" kwarg into the plotter call
    """

    identifier: str
    plotter: Callable
    stage: Union[str, Stage]
    targets: dict[str, str] = None
    figure_kwargs: dict[str] = None
    # TODO: iteration (global_step), or epoch?
    every_n: int = 1
    input_ax: bool = False

    def __post_init__(self):
        if isinstance(self.stage, str):
            self.stage = Stage(self.stage)

    def plot(self, **kwargs):
        figure = plt.figure(**self.figure_kwargs)
        # NOTE: Assume plotter plots to the current figure or takes in an ax
        if self.input_ax:
            self.plotter(ax=plt.gca(), **kwargs)
        else:
            self.plotter(**kwargs)
        plt.close()
        return figure


@delegates(ConfusionMatrixDisplay)
class ConfusionMatrixPlotter:
    """
    Plots a confusion matrix
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(
        self,
        confusion_matrix: Union[Tensor, np.ndarray],
        ax: Optional[Axes] = None,
    ):
        if isinstance(confusion_matrix, Tensor):
            confusion_matrix = confusion_matrix.to("cpu").numpy()
        confusion_matrix = np.around(confusion_matrix, 2)
        cm = ConfusionMatrixDisplay(confusion_matrix, **self.kwargs)

        if ax is None:
            ax = plt.gca()
        cm.plot(ax=ax)
