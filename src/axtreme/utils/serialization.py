"""Utilities for JSON serialization of axtreme objects.

Ax supports serialization of object to JSON, and deserialization from JSON back to objects, which is useful for saving experiments.
For this to work all parts of the object must be serializable/deserializable. This module provides helpers for this.
"""

from typing import Any


def serialize_not_deserializable(obj: Any) -> dict[str, Any]:
    """Serialize a _NotDeserialized object to a dict.

    ax.storage.json_store.decoder.object_from_json (which consume the output of this function) pops the __type key from
    the dict, So we need to have an additional key if we want to track the name of the class.
    """
    return {"__type": obj.__class__.__name__, "class_name": obj.__class__.__name__}  # pyright: ignore[reportGeneralTypeIssues]


class _NotDeserialized:
    """Placeholder for fields that are not yet supported by deserialization.

    For object that have not yet been serialized, create "exploding object" so operations that simply instantiate the
    object, but not use it can still be used. If anything tried to use the object, immediately return a
    NotImplementedError with context of what is not implemented. This is to avoid obscure AttributeError or NoneType
    error.

    Typically serialization will be done by `serialize_not_deserializable`.
    """

    def __init__(
        self,
        class_name: str,
        **kwargs: dict[str, Any],
    ) -> None:
        """Constructs the object.

        Args:
            class_name: The name of the class that was not deserialized. This is used to provide context in the
              NotImplementedError.
            **kwargs: Only so `__type` can be consumed, which makes the function compatible with how `object_from_json`
              might pass `serialize_not_deserializable` results (in some flows `__type` is stripped/included). If any
              other values are passed throws a value error.
        """
        if set(kwargs.keys()) - {"__type"}:
            raise ValueError(f"Unexpected kwargs {kwargs.keys()} passed to _NotDeserialized. Only `__type` is allowed.")

        object.__setattr__(self, "class_name", class_name)
        # Pass through methods can be called on the object.
        object.__setattr__(self, "_passthrough_methods", ["__deepcopy__"])

    def __getattr__(self, name: str) -> object:
        passthrough_methods = object.__getattribute__(self, "_passthrough_methods")
        if name in passthrough_methods:
            return object.__getattribute__(self, name)
        class_name = object.__getattribute__(self, "class_name")
        raise NotImplementedError(
            f"Deserialization of {class_name} is not implemented, and the method {name} is not allowed on this stub."
            "Serialization functions can be implemented (and potentially added to Ax registry "
            "from ax.storage.json_store.registry  import CORE_DECODER_REGISTRY, CORE_ENCODER_REGISTRY`)"
        )

    def __repr__(self) -> str:
        class_name = object.__getattribute__(self, "class_name")
        return f"_NotDeserialized({class_name})"
