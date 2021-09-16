from typing import Callable, Optional, Union

import matplotlib as mpl
from matplotlib.axes._axes import Axes
import matplotlib.pyplot as plt
from torch import nn
from torch import Tensor
from torchvision.transforms import Compose


# TODO: How to compose title with label?
#     * format title string using {variable}?
#     * Only add targets from visualblock?
#           * Will be easier to add correct label, dont need to propagate
#           sample_idx information.
#     * To open up for args and kwargs, add target_args and target_kwargs? ie.
#     if a plotting function does not kwargs? OR add mapping around the plotter
#     wo ensure kwargs.

# TODO: Plot take in multiple targets to 1. be able to combine them in the
# plotter and 2. make title with eg pred/true label

# TODO: How to plotting metrics per epoch. Need to accumulate eg. confusion
# matrix.


class Plot:
    def __init__(
        self,
        target: str,
        plotter: Union[Callable, str],
        plotter_kwargs: Optional[dict] = None,
        preprocessing: Optional[Union[nn.Module, Compose]] = None,
    ):
        self.target = target
        self.plotter = plotter
        self.plotter_kwargs = {} if plotter_kwargs is None else plotter_kwargs
        self.preprocessing = {} if preprocessing is None else preprocessing

    def __call__(self, data: dict[str, Tensor], ax: Axes):
        """
        :param data: batch data of shape (batch_size, channels, row, col)
        :param ax: axes to plot to
        """

        if self.target not in data:
            raise ValueError(f"{self.target} not in data, unable to plot")

        if isinstance(self.plotter, str):
            # If string assume plotter is a function called from the Axes object
            plotter = getattr(ax, self.plotter, None)
            if plotter is None:
                raise ValueError(f"{self.plotter} is not a valid plotter")
        else:
            plotter = self.plotter
            # NOTE: assumption that plotter class takes in ax object!
            self.plotter_kwargs["ax"] = ax

        # NOTE: Always plots first element in batch
        data_to_plot = data[self.target].detach().cpu()[0]
        data_to_plot = self.preprocessing(data_to_plot)
        ax.imshow(data_to_plot, **self.plotter_kwargs)


class Figure:
    """
    Represents a figure
    """

    def __init__(
        self,
        identifier: str,
        plot: Plot,
        title: Optional[str] = "",
        figure_kwargs: Optional[dict] = None,
    ):
        self.identifier = identifier
        self.title = title
        self.plot = plot
        self.figure_kwargs = {} if figure_kwargs is None else figure_kwargs
        self.target = plot.target

    def __call__(self, data: dict[str, Tensor]) -> mpl.figure.Figure:
        """
        :param data: batch data
        """
        figure = plt.figure(**self.figure_kwargs)

        plt.title(self.title)
        # TODO: search for {} in title and replace with sample_idx, target
        self.plot(data, plt.gca())
        return figure


class Subplots:
    """
    Represents subplots
    """

    def __init__(
        self,
        plots: list[Plot],
        subplots_kwargs: dict,
    ):
        self.plots = plots
        self.subplots_kwargs = subplots_kwargs

    def __call__(self, data: dict[str, Tensor]) -> mpl.figure.Figure:
        """
        data: data to plot
        kwargs: extra inputs to the plotting
        """
        figure, axes = plt.subplots(**self.subplots_kwargs)
        axes = axes.flatten()
        for plot, ax in zip(self.plots, axes):
            plot(data, ax)
        return figure


class VisualizationBlock:
    """
    Controls when to plot a figure
    """

    def __init__(
        self,
        plot,  # Plot to make
        stage: str,  # TODO: make into enum
        every_n_batch: int,
    ):
        self.plot = plot
        self.stage = stage
        self.every_n_batch = every_n_batch

    def __call__(
        self, data: dict[str, Tensor], batch_idx: int, step: int, logger
    ):
        """
        :param data: data from current batch.
        :param batch_idx: id of the current batch
        :param step_idx: id of current step
        :param logger: logger to add figure to
        """
        if self.every_n_batch and batch_idx % self.every_n_batch == 0:
            figure = self.plot(data)
            logger.add_figure(
                f"{self.stage}/{self.plot.identifier}", figure, step
            )
