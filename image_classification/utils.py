from functools import reduce
from typing import Union

from omegaconf import DictConfig


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
