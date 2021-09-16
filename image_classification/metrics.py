import torch
import torchmetrics


class MetricsWrapper(torchmetrics.Metric):
    def __init__(self, metric: torchmetrics.Metric, log: bool = True):
        """
        Wrapper around metrics. Adds some metainformation for easier controlling
        which are logged as scalars and which need special hendling. Example of
        a metrics that cannot be logged directly is the confusion matrics.
        """
        super().__init__()
        self.metric = metric
        self.log = log

    def update(self, prediction: torch.Tensor, label: torch.Tensor):
        self.metric.update(prediction, label)

    def reset(self):
        self.metric.reset()

    def compute(self):
        return self.metric.compute()
