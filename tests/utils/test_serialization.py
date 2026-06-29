import copy

import pytest

from axtreme.utils.serialization import _NotDeserialized


class TestNotDeserialized:
    def test_create_with_class_name_only(self):
        obj = _NotDeserialized(class_name="MyClass")
        assert object.__getattribute__(obj, "class_name") == "MyClass"

    def test_create_with_type_kwarg_ignored(self):
        obj = _NotDeserialized(class_name="MyClass", __type="MyClass")  # pyright: ignore[reportArgumentType]
        assert object.__getattribute__(obj, "class_name") == "MyClass"

    def test_unexpected_kwargs_raise_value_error(self):
        with pytest.raises(ValueError, match="Unexpected kwargs"):
            _NotDeserialized(class_name="MyClass", extra="bad")  # pyright: ignore[reportArgumentType]

    def test_attribute_access_raises_not_implemented(self):
        obj = _NotDeserialized(class_name="MyClass")
        with pytest.raises(NotImplementedError, match="MyClass"):
            _ = obj.some_attribute

    def test_pass_through_method_allowed(self):
        obj = _NotDeserialized(class_name="MyClass")
        assert isinstance(copy.deepcopy(obj), _NotDeserialized)  # pyright: ignore[reportCallIssue]
