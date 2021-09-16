from typing import Optional

import hydra
import torch


class FunctionTransform(torch.nn.Module):
    """
    Imports a method and applies it when called. This is convenient when reading
    configs from Hydra and using the instantiation functionality
    """

    def __init__(
        self, function_path: str, function_kwargs: Optional[dict] = None
    ):
        super().__init__()
        self.function = hydra.utils.get_method(function_path)
        self.function_kwargs = (
            {} if function_kwargs is None else function_kwargs
        )

    def forward(self, x: torch.Tensor):
        return self.function(x, **self.function_kwargs)


class LambdaTransform(torch.nn.Module):
    """
    Performs an arbitrary operation on a tensor.
    """

    def __init__(
        self, expression: str, imports: Optional[dict[str, str]] = None
    ):
        super().__init__()
        self.function = lambda x: eval(expression)

    def forward(self, x: torch.Tensor):
        return self.function(x)
