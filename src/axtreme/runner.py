"""Additional Runner implementatiions."""

from collections.abc import Iterable
from typing import Any

from ax import Runner, Trial
from ax.core.base_trial import BaseTrial, TrialStatus

from axtreme.evaluation import EvaluationFunction


class LocalMetadataRunner(Runner):
    """Except for the addition of evauation_function this is coppied exactly from ax.runners.SyntheticRunner."""

    def __init__(self, evaluation_function: EvaluationFunction) -> None:  # noqa: D107
        self.evaluation_function = evaluation_function

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
