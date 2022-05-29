from functools import reduce
from typing import Any, Callable, Mapping, Sequence, Union
import einops
import numpy as np

from omegaconf import DictConfig
from torch import Tensor
from torchmetrics import Metric


def map_recursive(obj: Union[dict, list, Any], func: Callable):
    """
    Map a function over the elements in a Sequence or the Values in a Mapping.
    """
    if isinstance(obj, Mapping):
        return {key: map_recursive(value, func) for key, value in obj.items()}
    elif isinstance(obj, Sequence):
        return [map_recursive(value, func) for value in obj]
    else:
        return func(obj)


def tensor2numpy(obj: Any):
    """
    Call detach(), cpu() and numpy() on all tensors in a sequence or mapping.
    """

    def func(x):
        if isinstance(x, Tensor):
            return x.detach().cpu().numpy()
        else:
            return x

    return map_recursive(obj, func)


def singlesequence2single(obj: Any):
    """
    If the length of a sequence is 1 or if it is a 0-dim array, replace the
    array by the single element.
    """

    def func(x):
        if isinstance(x, Tensor) and x.dim() == 0:
            return x.item()
        elif (
            isinstance(x, Sequence)
            or isinstance(x, Tensor)
            or isinstance(x, np.ndarray)
        ) and len(x) == 1:
            return x[0]
        return x

    return map_recursive(obj, func)


def select_in_dimension(obj: Any, idx: Union[Sequence[int], int]):
    # TODO: allow for skipping dimensions.
    """
    For each axis select an index. Does not support skipping selecting indices
    from a dimention.

    ie. idx = (0, 4) will return the 0 and 4 element in the two first
    dimensions.
    """
    if isinstance(idx, Sequence):
        idx = tuple(idx)

    def func(x):
        if isinstance(x, Tensor) or isinstance(x, np.ndarray):
            return x[idx]
        else:
            return x

    return map_recursive(obj, func)


def rearrange_tensors(obj: Any, pattern: str):
    """
    Rearrange all tensors/arrays in input according to input pattern.
    """

    def func(x):
        if isinstance(x, Tensor) or isinstance(x, np.ndarray):
            return einops.rearrange(x, pattern)
        else:
            return x

    return map_recursive(obj, func)


def compute_metrics(obj: Any) -> Any:
    """
    Compute a metric or metrics in a Sequence or Mapping.
    """

    def func(x):
        if isinstance(x, Metric):
            return x.compute()
        return x

    return map_recursive(obj, func)


def deep_get(dictionary: Union[dict, DictConfig], keys):
    """
    Access dictionary by nested keys.

    Assume we have a dict and a nested key defined as

    d = {"A": {"B": {"C": 10}} }
    keys = "A.B.C"

    then `deep_get(d, keys)` will return `10`
    """

    def is_dict(input):
        return isinstance(input, dict) or isinstance(input, DictConfig)

    return reduce(
        lambda d, key: d.get(key, None) if is_dict(d) else None,
        keys.split("."),
        dictionary,
    )


def simplify_search_space_key(key: str):
    """
    The search_space keys are nested keys that have all the keys to get from the
    global config into the value to be varied. This is not nice for
    visualization, hence clean the keys to one descriptive word.
    """

    key_split = key.split(".")
    # the last element is target when we search across different classes
    # that will be instantiated
    if key_split[-1] == "_target_":
        return key_split[-2]
    else:
        return key_split[-1]


def simplify_search_space_value(value: Union[str, int, float]):
    # TODO: If we are testing two different implementations a similarly named
    # class this breaks down.
    """
    If the value to search over is a _target_ string to be instantiated it
    contains the absolute import path to the class. In that case simplify the
    string into just the class name.
    """
    if isinstance(value, str):
        return value.split(".")[-1]
    else:
        return value


def args_kwargs_from_dict(data: dict[str], mapping: dict[str, str]):
    """
    Fetches the values from keys in a data dictionary and saves them to a new
    args at the assigned position or as kwargs with the assigned new key.
    """
    args = []
    kwargs = {}
    for key, new_key in mapping.items():
        if isinstance(new_key, int):
            args.append((new_key, data[key]))
        else:
            kwargs[new_key] = data[key]

    if args:
        _, args = zip(*sorted(args))

    return args, kwargs


def evaluate(expression: str):
    """
    Instantiating eval directly does not seem to twork with Hydra.
    """
    return eval(expression)
