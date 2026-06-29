"""Additional Runner implementatiions."""

from collections.abc import Iterable
from typing import Any

from ax import Runner, Trial
from ax.core.base_trial import BaseTrial, TrialStatus
from ax.storage.json_store.encoders import runner_to_dict
from ax.storage.json_store.registry import (
    CORE_DECODER_REGISTRY,
    CORE_ENCODER_REGISTRY,
)
from ax.utils.common.serialization import TClassDecoderRegistry, TDecoderRegistry

from axtreme.evaluation import EvaluationFunction
from axtreme.utils.serialization import _NotDeserialized, serialize_not_deserializable


class LocalMetadataRunner(Runner):
    """Except for the addition of evauation_function this is coppied exactly from ax.runners.SyntheticRunner."""

    def __init__(self, evaluation_function: EvaluationFunction) -> None:  # noqa: D107
        self.evaluation_function = evaluation_function

    @classmethod
    def serialize_init_args(cls, obj: Runner) -> dict[str, Any]:
        """Serialize the object so it can be saved to JSON.

        Note: This is typically used by `ax.storage.json_store.encoder.object_to_json` which drives the serialization.

        Returns:
        A dict of:
            str: The name of the parameter. Conceptually this should be an arg in the classes __init__ method, but so
             long as deserialize_init_args knows how to handle it, it can be anything.
            Any: This should be the object the parameter corresponses to. It does NOT have to be JSON serialized yet, as
             `object_to_json`  will recursively call encoders for the subcomponets.
        """
        # NOTE: While deserialize_init_args could just create the evaluation_function directly, this make the json more
        # readable and better conforms to the pattern used throughout ax.
        return {"evaluation_function": serialize_not_deserializable(obj.evaluation_function)}

    @classmethod
    def deserialize_init_args(
        cls,
        args: dict[str, Any],
        decoder_registry: TDecoderRegistry | None = None,  # noqa: ARG003
        class_decoder_registry: TClassDecoderRegistry | None = None,  # noqa: ARG003
    ) -> dict[str, Any]:
        """Deserialize the init args from JSON (which can then be used to create the object cls(**output)).

        Note: this is typically used by `ax.storage.json_store.decoder.object_from_json` which drives the
        deserialization.

        Args:
            args: This is the output of serialize_init_args, but with all subcomponents already serialized (
              `serialize_init_args` can return raw python objects, but all inputs here are expected to be JSON
               serialized).
            decoder_registry: `object_from_json` which drives the deserialization will typically pass
              `ax.storage.json_store.registry.CORE_DECODER_REGISTRY`
            class_decoder_registry: `object_from_json` which drives the deserialization will typically pass
              `ax.storage.json_store.registry.CORE_CLASS_DECODER_REGISTRY`

        Returns: A dict of:
            str: The name of parameter to be passed to the classes __init__ method.
            Any: Instantiated object to pass. NOTE: `object_from_json` doesn't do any further recursive deserialization
              to subcomponents, output is directly sed to instantiate the object.
        """
        # Could also used _NotDeserialized('EvaluationFunction') below directly.
        return {"evaluation_function": _NotDeserialized(**args["evaluation_function"])}

    def run(self, trial: BaseTrial) -> dict[str, Any]:  # noqa: D102
        # The other available type is BatchTrail which is for very specific usecase not relevant to this project. see https://ax.dev/docs/core.html#trial-vs-batched-trial
        if isinstance(trial, Trial):
            deployed_name = (
                f"{trial.experiment.name}_{trial.index!s}" if trial.experiment.has_name else str(trial.index)
            )
            assert trial.arm is not None  # primarily to deal with typing
            simulation_result = self.evaluation_function(trial.arm.parameters)

            return {"name": deployed_name, "simulation_result": simulation_result}

        msg = f"Currently only trial of type Trial is supported. Received {type(trial)}."
        raise NotImplementedError(msg)

    def poll_trial_status(self, trials: Iterable[BaseTrial]) -> dict[TrialStatus, set[int]]:  # noqa: D102
        # only completed if simulation_result is available
        completed_trials = [t for t in trials if t.run_metadata.get("simulation_result") is not None]
        # TODO(magkri): not sure how to handle all TrialStatus values here, maybe we shouldn't override this method
        return {
            TrialStatus.COMPLETED: {t.index for t in completed_trials},
            TrialStatus.RUNNING: {t.index for t in trials if t not in completed_trials},
        }

    @property
    def run_metadata_report_keys(self) -> list[str]:  # noqa: D102
        return ["name", "simulation_result"]


# Add to Ax's global encoder/decoder registry so LocalMetadataRunner can automatically be saved
CORE_ENCODER_REGISTRY[LocalMetadataRunner] = runner_to_dict
CORE_DECODER_REGISTRY["LocalMetadataRunner"] = LocalMetadataRunner
