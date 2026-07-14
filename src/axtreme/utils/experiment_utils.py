"""Helpers for working with the Ax Experiment class."""

from typing import TypeAlias, TypedDict, cast

import pandas as pd
from ax import Experiment

from axtreme.experiment import add_metric_data_to_experiment


def input_out_df_from_experiment(exp: Experiment) -> pd.DataFrame:
    """Shows the x and y data in one dataframe."""
    df_y = exp.fetch_data().df
    df_x = pd.DataFrame([{**trial.arm.parameters, "arm_name": trial.arm.name} for trial in exp.trials.values()])  # pyright: ignore[reportAttributeAccessIssue]
    return df_y.merge(df_x, on="arm_name", how="inner")


class _MetricValueData(TypedDict):
    mean: float | None
    sem: float | None


# Key is the parameter name
_ParametersData: TypeAlias = dict[str, None | str | bool | float | int]
# Key is the metric name
_MetricData: TypeAlias = dict[str, _MetricValueData]


class _ArmData(TypedDict):
    parameters: _ParametersData
    metrics: _MetricData


# ExperimentRunsData stores the details (x location, result) of all the completed trials an experiment has run.
# Structure: dict[<trial_index_str>, dict[<arm_name>, _ArmData]]
# NOTE: We have created this data structure for our own convenience. It is not part of or related to the ax library.
# TODO(sw 2026-06-14): This could be upgraded to use Ax types (e.g. axtreme.core.types.TParameterization)
ExperimentRunsData: TypeAlias = dict[str, dict[str, _ArmData]]


def extract_data_from_experiment_as_json(
    experiment: Experiment,
) -> ExperimentRunsData:
    """Extracts the details (x location, results) of the completed trials of an Ax Experiment.

    This does not serialize the full Experiment. The information extracted is the minimum necessary for these trial runs
    to be added to another experiment.

    Args:
        experiment: The Ax experiment to extract the data from.

    Returns:
        A dictionary with the following structure:

        .. code-block:: json

            {
                <trial_index_str>: {
                    "<arm_name>": {
                        "parameters": {
                            "<parameter_name>": <parameter_value>
                        },
                        "metrics": {
                            "<metric_name>": {
                                "mean": 0.0,
                                "sem": 0.0
                            }
                        }
                    }
                }
            }
    """
    metric_data = experiment.fetch_data().df
    trial_indexes = metric_data[["trial_index"]].drop_duplicates()
    trials = experiment.get_trials_by_indices(trial_indexes["trial_index"].values)

    trial_data: ExperimentRunsData = {}

    metric_data_by_arm = metric_data.groupby("arm_name")

    for trial in trials:
        trial_index = str(trial.index)
        if trial_index not in trial_data:
            # Keys in json dictionaries must be strings.
            trial_data[trial_index] = {}
        # BatchTrails can have multiple arms
        for arm in trial.arms:
            if arm.name not in trial_data[trial_index]:
                trial_data[trial_index][arm.name] = {
                    "parameters": {},
                    "metrics": {},
                }

            for parameter_name, parameter_value in arm.parameters.items():
                trial_data[trial_index][arm.name]["parameters"][parameter_name] = parameter_value

            arm_metric_data = metric_data_by_arm.get_group(arm.name)[["metric_name", "mean", "sem"]]
            for metric_name, mean, sem in arm_metric_data.to_numpy():
                trial_data[trial_index][arm.name]["metrics"][metric_name] = {"mean": mean, "sem": sem}

    return trial_data


def add_json_data_to_experiment(experiment: Experiment, json_data: ExperimentRunsData) -> None:
    """Adds the data from a dictionary to an ax Experiment.

    Args:
        experiment: The ax Experiment to add the data to.
        json_data: The data to add to the ax Experiment. The structure should be the same as the output of
            extract_data_from_experiment.

    Example:
        .. code-block:: json

            {
                <trial_index_str>: {
                    "<arm_name>": {
                        "parameters": {
                            "<parameter_name>": <parameter_value>
                        },
                        "metrics": {
                            "<metric_name>": {
                                "mean": 0.0,
                                "sem": 0.0
                            }
                        }
                    }
                }
            }
    """
    # For each arm in the dictionary, extract the parameters and metrics for all arms in the trial
    for trial_arms in json_data.values():
        # If a single trial has multiple arms, a batch trial is created.
        trial_parameters: list[_ParametersData] = [arm_data["parameters"] for arm_data in trial_arms.values()]

        # add_metric_data_to_experiment's metric_data param is a broader union (it also accepts float/tuple
        # forms from other callers) than the _MetricValueData TypedDict -- cast to bridge the two.
        trial_metrics: list[dict[str, dict[str, float | None]]] = [
            cast("dict[str, dict[str, float | None]]", arm_data["metrics"]) for arm_data in trial_arms.values()
        ]
        _ = add_metric_data_to_experiment(
            experiment=experiment,
            parameterizations=trial_parameters,
            metric_data=trial_metrics,
        )
