from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence, Union

from torch import Tensor
from torch.nn import Module

from image_classification import Stage


# TODO: Add possibility to detach before transforming.
@dataclass
class Transform:
    """
    Container for transforms. Should always apply to the entire batch. The
    resulting tensors will be returned in a dictionary where the key is on the
    form "<target>_<identifier>".
    """

    identifier: str
    transform: Sequence[Union[Module, Callable]]
    targets: Sequence[str]
    stage: Union[str, Stage]
    every_n: Optional[int]
    check_batch_size: bool = True

    def __post_init__(self):
        if isinstance(self.stage, str):
            self.stage = Stage(self.stage)

        self.postfix = self.identifier.strip().replace(" ", "-").lower()
        if self.postfix != "":
            self.postfix = "_" + self.postfix

        if isinstance(self.targets, str):
            self.targets = [self.targets]

    def _check_batch_size(
        self, original: Tensor, transformed: Tensor, target: str
    ):
        batch_size = len(original)
        if transformed.dim() == 0 or len(transformed) != len(original):
            raise RuntimeError(
                f"Transform '{self.identifier}' does not return "
                f"correct number of samples on target '{target}'! "
                f"Output shape {transformed.shape} does not match batch_size "
                f"{batch_size}"
            )

    def __call__(self, data: dict[str]):
        result = {}
        for target in self.targets:
            result_key = target + self.postfix
            result[result_key] = self.transform(data[target])

            if self.check_batch_size:
                self._check_batch_size(data[target], result[result_key], target)

        return result


@dataclass
class TransformScheduler:
    """
    Handle which transforms should be called at which stage and step.
    """

    transforms: Union[Sequence[Transform], Transform]

    def __post_init__(self):
        if not isinstance(self.transforms, Sequence):
            self.transforms = [self.transforms]

    def __call__(
        self,
        data: dict[str, Any],
        stage: Stage,
        step: int,
    ):
        results = {}
        for transform in self.transforms:
            if transform.stage == stage and step % transform.every_n == 0:
                results = results | transform(data)
        return results
