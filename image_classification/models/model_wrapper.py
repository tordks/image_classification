from typing import Optional, Union

import pytorch_lightning as pl
from torch import Tensor
from torch.nn import Module


class ModelWrapper(pl.LightningModule):
    def __init__(
        self,
        network: Module,
        transform: Optional[Module] = None,
        padding: int = 0,
        target: str = "feature",
    ):
        super().__init__()
        self.network = self.prepare_network(network, transform, padding)
        self.target = target

    def prepare_network(
        self,
        network: Module,
        transform: Module,
        padding: int,
    ):
        # TODO: padding
        # TODO: transforms
        return network

    def forward(self, sample: Union[dict[str], Tensor]):
        if isinstance(sample, Tensor):
            feature = sample
        elif isinstance(sample, dict):
            feature = sample[self.target]

        return self.network(feature)
