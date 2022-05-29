import re
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence, Union
import einops

import matplotlib.pyplot as plt
import numpy as np
from fastcore.meta import delegates
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sklearn.metrics import ConfusionMatrixDisplay
from torch import Tensor

from image_classification import Stage, logger
from image_classification.utils import (
    args_kwargs_from_dict,
    compute_metrics,
    select_in_dimension,
    singlesequence2single,
    tensor2numpy,
    rearrange_tensors,
)


@dataclass
class Plotter:
    """
    Container for a plotting callable.

    :param plotter: Callable which plots to a matplotlib figure.
    :param targets: Mapping from the data key within the training pipeline to
                    the name of the kwarg in the plotter.
    :param tensor_idx: idx to select from each axis of all tensor/array in
                       targets
    :param rearrange: einops rearrange pattern to be applied to matching . Can
                      either be one string and in that case the rearrange is
                      applied to all tensors in targets. Otherwise it is a dict
                      with target key to pattern mapping. See
                      https://einops.rocks/api/rearrange/ for syntax.
    """

    plotter: Callable
    targets: dict[str, str]
    tensor_idx: Optional[Union[int, list[int]]] = None
    rearrange_pattern: Optional[Union[dict[str, str], str]] = None

    def rearrange(self, args: list, kwargs: dict[str]):
        """
        rearrange args and kwargs according to given pattern
        """
        if isinstance(self.rearrange_pattern, str):
            args = rearrange_tensors(args, self.rearrange_pattern)
            kwargs = rearrange_tensors(kwargs, self.rearrange_pattern)
        elif isinstance(self.rearrange_pattern, dict):
            for key, pattern in self.rearrange_pattern.items():
                if isinstance(key, int):
                    args[key] = einops.rearrange(args[key], pattern)
                elif isinstance(key, str):
                    kwargs[key] = einops.rearrange(kwargs[key], pattern)
                else:
                    raise ValueError(
                        f"key should be 'str' or 'int', not {type(key)}"
                    )
        return args, kwargs

    def plot(self, *args, **kwargs):
        if self.tensor_idx is not None:
            args = select_in_dimension(args, self.tensor_idx)
            kwargs = select_in_dimension(kwargs, self.tensor_idx)

        args, kwargs = self.rearrange(args, kwargs)

        self.plotter(*args, **kwargs)


# TODO: Explicitly define interface, so that it will be easier to implement
# other figure classes
# TODO: How to handle a plotter that takes in a batch and plots information from
# it? Create BatchSubplots class? Plotter that adds subplots to a figure?
@dataclass
class Subplots:
    """
    A figure of subplots with associated plotters. Holds information on which
    arguments goes into the plotters and at which stage in the training pipeline
    a plot should be made.

    It also makes a mapping of the input data to the data needed by the plotter.
    In addition to the input data it makes available the current axis to the
    plotter.

    :param identifier: Descriptive identification of figure
    :param stage: The stage at which to perform the plotting
    :param every_n: Plot every nth step. The meaning step changes depending on
                    which stage the plot happens for. if stage =
                    "on_train_epoch_end", then step is interpretted as epoch
                    number
    :param title: Title to give figure. Defaults to identifier. Supports
                  formatting using elements from the input data dict. The
                  identifier is also available in the formatting.
    :param subplots_kwargs: kwargs which goes into the
                            matplotlib.pyplot.subplots call
    """

    # TODO: Consider allowing for multiple stages for the same figure
    #  * stage -> stages
    #  * need to specify target per stage => target -> dict[Stage, targets]
    #  * accept patterns? eg. both val_cm and train_cm
    identifier: str
    plotters: list[Plotter]
    stage: Union[str, Stage]
    every_n: int
    nrows: int = 1
    ncols: int = 1
    batch_idx: Optional[int] = None
    title: Optional[str] = None
    subplots_kwargs: dict[str] = None

    def __post_init__(self):
        if isinstance(self.stage, str):
            self.stage = Stage(self.stage)

        if self.title is None:
            self.title = self.identifier

        if self.nrows * self.ncols < len(self.plotters):
            raise ValueError(
                "The number of plotters greater than amount of subplots."
            )
        layout = {"nrows": self.nrows, "ncols": self.ncols}
        if self.subplots_kwargs is None:
            self.subplots_kwargs = layout
        else:
            self.subplots_kwargs = self.subplots_kwargs | layout

    def _prepare_args_or_kwargs(
        self, args_or_kwargs: Union[list, dict]
    ) -> Union[list, dict]:
        """
        Operations that needs to be performed on all args and kwargs before any
        more computation occurs.
        """
        args_or_kwargs = compute_metrics(args_or_kwargs)
        args_or_kwargs = tensor2numpy(args_or_kwargs)
        args_or_kwargs = singlesequence2single(args_or_kwargs)

        if self.batch_idx is not None:
            args_or_kwargs = select_in_dimension(args_or_kwargs, self.batch_idx)

        return args_or_kwargs

    def prepare(self, data: dict[str], args_kwargs_mapping: dict[str, str]):
        """
        Prepare args and kwargs that will be put into the plotting call.
        """

        plot_args, plot_kwargs = args_kwargs_from_dict(
            data, args_kwargs_mapping
        )
        plot_args = self._prepare_args_or_kwargs(plot_args)
        plot_kwargs = self._prepare_args_or_kwargs(plot_kwargs)
        return plot_args, plot_kwargs

    def format_title(self, data: dict[str]) -> str:
        """
        Formats title using input data.
        """
        title_targets = {
            key: key for key in re.findall(r"\{(.*?)\}", self.title)
        }
        _, title_kwargs = self.prepare(data, title_targets)
        title = self.title.format(
            **(title_kwargs | {"identifier": self.identifier})
        )
        return title

    def plot(self, data: dict[str]) -> Figure:
        """
        Create figure and perform the specified plotting.
        """

        figure, axes = plt.subplots(**self.subplots_kwargs)
        if not isinstance(axes, Sequence):
            axes = np.array([axes])
        axes = axes.flatten()

        logger.info(f"Plot '{self.identifier}' at stage '{self.stage.value}'")

        for ax, plotter in zip(axes, self.plotters):
            logger.info(plotter)
            plt.sca(ax)
            plot_args, plot_kwargs = self.prepare(
                data | {"ax": ax}, plotter.targets
            )
            plotter.plot(*plot_args, **plot_kwargs)

        plt.title(self.format_title(data))
        plt.close()
        return figure


@dataclass
class VisualizationScheduler:
    """
    Handle which figures should be visualized at which stage and step.
    """

    figures: Union[Sequence[Subplots], Subplots]

    def __post_init__(self):
        if not isinstance(self.figures, Sequence):
            self.figures = [self.figures]

    def __call__(
        self,
        data: dict[str, Any],
        stage: Stage,
        step: int,
    ):
        figures_to_plot = {}
        for figure in self.figures:
            if figure.stage == stage and step % figure.every_n == 0:
                figures_to_plot[figure.identifier] = figure.plot(data)
        return figures_to_plot


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
