import json

import pytest
from ax.storage.json_store.decoder import object_from_json
from ax.storage.json_store.encoder import object_to_json
from ax.storage.json_store.encoders import runner_to_dict

from axtreme.evaluation import EvaluationFunction
from axtreme.runner import LocalMetadataRunner
from axtreme.utils.serialization import _NotDeserialized


def test_local_metadata_runner_serialization_manual():
    """Test that LocalMetadataRunner can be serialized.

    NOTE: This test will need to be updated if any change to `serialize_init_args` or `deserialize_init_args` is made.

    """
    # Dummy object for test
    eval_func = EvaluationFunction(None, None, None, None)  # pyright: ignore[reportArgumentType]
    runner = LocalMetadataRunner(evaluation_function=eval_func)

    runner_serialized = LocalMetadataRunner.serialize_init_args(
        runner
    )  # This will use the custom serialization for LocalMetadataRunner

    expected_serialization = {
        "evaluation_function": {"__type": "EvaluationFunction", "class_name": "EvaluationFunction"}
    }

    assert runner_serialized == expected_serialization


def test_local_metadata_runner_deserialization_manual():
    """Test that LocalMetadataRunner can be deserialized.

    NOTE: This test will need to be updated if any change to `serialize_init_args` or `deserialize_init_args` is made.
    """

    serialized_runner = {
        "__type": "LocalMetadataRunner",
        "evaluation_function": {"__type": "EvaluationFunction", "class_name": "EvaluationFunction"},
    }

    _ = serialized_runner.pop("__type", None)  # Remove the __type key to simulate what object_from_json does
    deserialized_args = LocalMetadataRunner.deserialize_init_args(serialized_runner)
    deserialized_runner = LocalMetadataRunner(**deserialized_args)

    # Check that the deserialized runner has the same evaluation function class name
    assert isinstance(deserialized_runner.evaluation_function, _NotDeserialized)
    assert deserialized_runner.evaluation_function.class_name == "EvaluationFunction"


@pytest.mark.integration
def test_local_metadata_runner_to_and_from_json():
    """Check successful integration the the function the typically drive this process."""
    # Dummy object for test
    eval_func = EvaluationFunction(None, None, None, None)  # pyright: ignore[reportArgumentType]
    runner = LocalMetadataRunner(evaluation_function=eval_func)

    encoder = {LocalMetadataRunner: runner_to_dict}
    decoder = {"LocalMetadataRunner": LocalMetadataRunner}

    runner_as_json = object_to_json(runner, encoder_registry=encoder)
    try:
        json.dumps(runner_as_json)  # Check if the serialized output is valid JSON
    except TypeError as e:
        pytest.fail(f"Serialized runner is not valid JSON: {e}")

    runner_deserialized = object_from_json(runner_as_json, decoder_registry=decoder)

    assert isinstance(runner_deserialized.evaluation_function, _NotDeserialized)
    assert runner_deserialized.evaluation_function.class_name == "EvaluationFunction"
