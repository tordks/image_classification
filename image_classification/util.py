from importlib import import_module
from typing import Callable, Union, cast
import sys  # only for defining ModuleType

from loguru import logger


ModuleType = type(sys)


def dynamic_import(name: str) -> Union[Callable, ModuleType]:
    """
    Dynamically imports a module or an attribute.
    """
    logger.info(f"importing {name}")

    try:
        name_parts = name.split(".")
        module_name = ".".join(name_parts[0:-1])
        attribute_name = name_parts[-1]

        module = import_module(module_name)

        if hasattr(module, attribute_name):
            imported = getattr(module, attribute_name)
        else:
            imported = module

    except Exception as err:
        logger.error(f"Could not import {name}")
        raise err

    return imported


def dynamic_loader(config: dict, extra_args=None, extra_kwargs=None, call=True):
    """
    config is on the format:
    {
        name: module.<>.<>
        args: [...]
        kwargs: {...}
    }
    """
    # TODO: open up for recursive loading. ie. if kwargs contains name, args and
    #       kwargs.
    # TODO: allow for partial call?
    imported_attr = dynamic_import(config["name"])
    if not callable(imported_attr):
        raise TypeError("Can only load callables")
    imported_attr = cast(Callable, imported_attr)  # silence type checker

    args = config.get("args", [])
    if extra_args is not None:
        args = [*args, *extra_args]
    kwargs = config.get("kwargs", {})
    if extra_kwargs is not None:
        kwargs = {**kwargs, **extra_kwargs}

    if call:
        return imported_attr(*args, **kwargs)
    else:
        return imported_attr
