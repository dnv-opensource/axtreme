"""Helpers for unpacking nestes objects for logging purposes.

When running experiments it is useful to record the objects that produced those results for reproducibility. This module
provides helper for unpacking nested objects for logging. It default implementations, and supports cusomistion so the
right level of granuality can be achieved.
"""

# pyright: reportUnnecessaryTypeIgnoreComment=false

# %%
from collections.abc import Callable
from typing import Any, TypeAlias, TypeVar, Union

import torch

T = TypeVar("T")
# Nested dict where keys are alsways strings
NestedDict: TypeAlias = dict[str, Union[Any, "NestedDict"]]
NestedStrDict: TypeAlias = dict[str, Union[str, "NestedStrDict"]]


# %%
def unpack_object(
    obj: object,
    custom_unpacking_config: dict[type, Callable[[Any], NestedDict]] | None = None,
    depth: int = 1,
) -> NestedDict:
    """Recursively extracts attributes from objects.

    This can be useful for logging the state of an object. It will unpack the public attributes of an object up to
    'depth'. Specific unpacking function can also be provided for attribute object in `custom_unpacking_config`. If
    these objects are encounter the unpacking function will be used instead.

    Args:
        obj: the object to unpack.
        custom_unpacking_config: Overrides default unpacking behaviour for object that subclass the keys.

            - Keys: Types
            - Values: Functions that take instance of that type, and produces a custom unpacking.
            - This unpacking should be of the following format:

        depth:
            - How many levels of objects to unpack.

    Return:
        A nested dictionary.

        - Without custom unpacking:
            - keys: are the public attribute names
            - values: are the attribute value, or nested dictionary of the object being unpacked.

                {"__class__": RootObjectClass, "attribute_1": {"__class__": Foo, "a": "blah", "b": None}, "attribute_2": "x"}

        - With custom unpacking:
            - keys: Determined by the custom unpacking function if used otherwise as above.
            - Values: Determined by the custom unpacking function if used otherwise as above.

                {"__class__": RootObjectClass, "attribute_1": {"custom_unpacking_key1": "name of class if FOO", "custom_unpacking_key2": [1, 2, 3]}, "attribute_2": "x"}

    Todo:
        depth: How many levels of objects to unpack.
    """  # noqa: E501
    custom_unpacking_config = custom_unpacking_config or {}

    if item_unpacker := get_closest_class_value(obj, custom_unpacking_config):
        return item_unpacker(obj)
    # Base case: out of depth - not further processing
    if depth == 0:
        return {"__class__": type(obj)}
    # Basecase: if unpackable object do nothing
    if not hasattr(obj, "__dict__"):
        return {"__class__": type(obj)}

    # Recussion:
    # - find child items
    # - work out which ones can be processed
    attributes = public_vars(obj)

    # Process if: tthe child can be unpacked.
    # sourcery skip: dict-comprehension
    updated_attribute: dict[str, Any] = {}
    for attr_name, attr_value in attributes.items():
        # Don't process
        if type(attr_value) is type:
            updated_attribute[attr_name] = attr_value
        # Check if can be expanded or there is a specific processing function.
        elif hasattr(attr_value, "__dict__") or get_closest_class_value(attr_value, custom_unpacking_config):  # type: ignore[arg-type]
            updated_attribute[attr_name] = unpack_object(attr_value, custom_unpacking_config, depth=depth - 1)

    return {"__class__": type(obj), **attributes, **updated_attribute}


def get_closest_class_value(obj: object, dic: dict[type, T]) -> T | None:
    """Searches a dictionary for the closest class, and return the value stored.

    Searches the class heirachy from bottom to top for a matching key in the dictionary. Returns the values stored with
    by that key.

    Args:
        obj: Object to find the class for
        dic: takes object of that type,

    Return:
        The value in stored in the dictionary for the closest class, or None if no match is found.
    """
    return next((dic[cls] for cls in obj.__class__.mro() if cls in dic), None)


def public_vars(obj: object) -> dict[str, Any]:
    """Like `vars()` but just returns public items."""
    return {k: v for k, v in vars(obj).items() if (k[0] != "_")}


def nested_content_to_str(d: NestedDict) -> NestedStrDict:
    """Helper function to turn all values in nested dictionaries to `str`."""

    def convert(value: Any) -> NestedDict | str:  # noqa: ANN401
        # If the value is a dictionary, recursively apply the function
        if isinstance(value, dict):
            return {str(k): convert(v) for k, v in value.items()}
        return str(value)

    return convert(d)  # type: ignore  # noqa: PGH003


def unpack_object_str_content(
    obj: object, custom_unpacking_config: None | dict[type, Callable[[Any], NestedDict]] = None, depth: int = 1
) -> NestedStrDict:
    """Helper that converts all `unpack_object` to string.

    Args:
        obj: the object to unpack.
        custom_unpacking_config: Overrides default unpacking behaviour for object that subclass the keys. If `None`
            `default_config()` is used.

            - Keys: Types
            - Values: Functions that take instance of that type, and produces a custom unpacking. This unpacking should
              be of the following format:

        depth:
            - How many levels of objects to unpack.
    """
    if custom_unpacking_config is None:
        custom_unpacking_config = default_config()
    return nested_content_to_str(unpack_object(obj, custom_unpacking_config, depth))


def default_config() -> dict[type, Callable[[Any], NestedDict]]:
    """Default/handy processing configurations."""
    import numpy as np
    from botorch.models import SingleTaskGP

    def process_single_task_gp(x: SingleTaskGP) -> dict[str, Any]:
        return {
            "__class__": type(x),
            "train_inputs": x.train_inputs[0].shape,  # pyright: ignore[reportOptionalSubscript]
            "train_targets": x.train_targets.shape,
            "likelihood.noise": x.likelihood.noise.shape,
        }

    # Helper for producing output
    def class_and_shape(x):  # type: ignore  # noqa: ANN001, ANN202, PGH003
        return {"__class__": type(x), "shape": x.shape}

    return {torch.Tensor: class_and_shape, np.ndarray: class_and_shape, SingleTaskGP: process_single_task_gp}
