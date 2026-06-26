"""Utilities for JSON serialization of axtreme objects.

Ax supports serialization of object to JSON, and deserialization from JSON back to objects, which is useful for saving experiments.
For this to work all parts of the object must be serializable/deserializable. This module provides helpers for this.
"""

from typing import Any, Never


def serialize_not_deserializable(obj: Any) -> dict[str, Any]:
    """Serialize a _NotDeserialized object to a dict.

    ax.storage.json_store.decoder.object_from_json (which consume the output of this function) pops the __type key from
    the dict, So we need to have an additional key if we want to track the name of the class.
    """
    return {"__type": obj.__class__.__name__, "class_name": obj.__class__.__name__}  # pyright: ignore[reportGeneralTypeIssues]


class _NotDeserialized:
    """Placeholder for fields that are not yet supported by deserialization.

    For object that have not yet been serialized, create "exploding object" so operations that simply instantiate the
    object, but not use it can still be used. If anything tried to use the object, immediately return a NotImplementedError
    with context of what is not implemented. This is to avoid obscure AttributeError or NoneType error.

    Typically serialization will be done by `serialize_not_deserializable`
    """

    def __init__(self, class_name: str, __type: str | None = None) -> None:
        """Constructs the object.

        Args:
            class_name: The name of the class that was not deserialized. This is used to provide context in the
              NotImplementedError.
            __type: This argument is not used, make the function compatible with how `object_from_json` might pass
              `serialize_not_deserializable` results (in some flows `__type` is stripped/included)
        """
        object.__setattr__(self, "class_name", class_name)

    def __getattr__(self, name: str) -> Never:
        class_name = object.__getattribute__(self, "class_name")
        raise NotImplementedError(
            f"Deserialization of {class_name} is not implemented "
            "(iteration 1 stub). Implement deserialize_init_args to support a runnable experiment."
        )

    def __repr__(self) -> str:
        class_name = object.__getattribute__(self, "class_name")
        return f"_NotDeserialized({class_name})"
