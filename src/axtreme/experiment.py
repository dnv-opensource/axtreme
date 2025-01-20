"""Helper functions for ax Experiments."""

from collections.abc import Iterable, Mapping
from typing import Any, cast

import numpy as np
import pandas as pd
from ax import Data, Experiment, SearchSpace
from ax.core.objective import MultiObjective, Objective
from ax.core.optimization_config import MultiObjectiveOptimizationConfig
from ax.core.types import TParameterization
from ax.modelbridge import Models
from numpy.typing import NDArray
from scipy.stats import rv_continuous

from axtreme.evaluation import EvaluationFunction
from axtreme.metrics import LocalMetadataMetric
from axtreme.runner import LocalMetadataRunner
from axtreme.simulator import Simulator
from axtreme.utils.distibution_helpers import distribution_parameter_names_from_scipy


def make_experiment(
    simulator: Simulator,
    search_space: SearchSpace,
    dist: rv_continuous,
    n_simulations_per_point: int = 100,
) -> Experiment:
    """Returns an experiment according to the given simulator, search_space, and dist_class.

    Args:
        simulator: The simulator to use for the experiment.
        search_space: The ax search space to use for the experiment.
        dist: The distribution that the result of a simulation is assumed to follow.
        n_simulations_per_point: The number of simulations to run for each point in the experiment.

    Returns:
        An ax Experiment.
    """
    # Define the metrics from the chosen distibution
    metrics = [
        LocalMetadataMetric(name=param, lower_is_better=False)
        for param in distribution_parameter_names_from_scipy(dist)
    ]
    # Optimisation config follows directly from the metrics
    mo = MultiObjective(
        objectives=[Objective(metric=m) for m in metrics],
    )
    optimization_config = MultiObjectiveOptimizationConfig(
        objective=mo,
        # TODO(sw): work out if we should provide these
        # Objective_threshold. They are only really needed when calulating the pareto_frontier.
    )

    # Construst runner config.
    # NOTE: This is completely determined by simulator and distibution
    # TODO(sw): issue #42 n_simulations_per_point: Need pick this is a more realistic way
    eval_function = EvaluationFunction(
        simulator=simulator,
        output_dist=dist,
        parameter_order=list(search_space.parameters.keys()),
        n_simulations_per_point=n_simulations_per_point,
    )
    runner = LocalMetadataRunner(eval_function)

    return Experiment(search_space=search_space, optimization_config=optimization_config, runner=runner)


def add_sobol_points_to_experiment(experiment: Experiment, n_iter: int = 5, seed: int | None = None) -> None:
    """Adds some points (chosen by sobol) to an experiment.

    Typically used to initialise the experiment so a GP has data for training.
    """
    # TODO(sw): this should deep copy the experiment,
    #           returning the new experiment rather than modifying an object in place?

    sobol = Models.SOBOL(search_space=experiment.search_space, seed=seed)
    for _ in range(n_iter):
        trial = experiment.new_trial(sobol.gen(1))
        _ = trial.run()
        _ = trial.mark_completed()


def add_metric_data_to_experiment(
    experiment: Experiment,
    parameterizations: Iterable[TParameterization],
    metric_data: Iterable[Mapping[str, float | tuple[float, float | None] | dict[str, float | None]]],
) -> tuple[Data, int]:
    """Add metric data to an experiment.

    This function is used to add data from previously run simulations to an experiment.
    To use this function one needs to have an estimate of the mean and standard error for each metric.
    This is useful if the simulator is slow or expensive to run and the data is already available.

    Args:
        experiment: The ax Experiment to add the data to.
        parameterizations: The parameterizations used in the simulation.

            - Iterable where each element is a dict parameter names and values ({[param_name]: value}).

        metric_data: The metric data from the simulation.

            - Iterable where each element is a dict of metric names and values.
            - The values can be a float, a tuple of (mean, sem), or a dict of {"mean": mean, "sem": sem}.
            - If a float is provided, the sem is assumed to be 0.

    Returns:
        - The data that was attached to the experiment.
        - The trial index of the trial that was created.
    """
    # Attach all the parameterizations to the trial (one arm per parameterization)

    # First check that all parameterizations are in the search space
    # Only attach the parameterizations that are in the search space
    # Only use the metric data for the parameterizations that are in the search space
    validated_parameterizations: list[TParameterization] = []
    validated_metric_data: list[Mapping[str, float | tuple[float, float | None] | Mapping[str, float | None]]] = []
    for parameterization, metric_data_for_param in zip(parameterizations, metric_data, strict=False):
        try:
            experiment.search_space.validate_membership(parameterization)
            validated_parameterizations.append(parameterization)
            validated_metric_data.append(metric_data_for_param)
        except ValueError as e:
            msg = f"Parameterization {parameterization} is not in the search space.\n\t{e!s}"
            print(msg)  # noqa: T201

    arms, trial_index = experiment.attach_trial(parameterizations=validated_parameterizations)

    # Creating a pandas DataFrame to store the output data for all the arms
    rows = []
    for arm_name, metric_result in zip(arms, validated_metric_data, strict=False):
        # Adding a row for each metric
        for metric_name in experiment.metrics:
            # Get mean and sem from the distribution result
            metric_mean_sem = metric_result[metric_name]
            mean: float | None = None
            sem: float | None = None
            if isinstance(metric_mean_sem, dict):
                mean = metric_mean_sem["mean"]
                sem = metric_mean_sem["sem"]
            elif isinstance(metric_mean_sem, tuple):
                mean, sem = metric_mean_sem
            elif isinstance(metric_mean_sem, float):
                mean = metric_mean_sem
            else:
                msg = f"Expected metric_data to be of type float, tuple or dict, got: {type(metric_mean_sem)}"
                raise TypeError(msg)

            assert mean is not None, f"Expected mean of metric {metric_name} of {arm_name} to be a float, got None"
            # Append the resulting new row
            new_row = {
                "arm_name": arm_name,
                "metric_name": metric_name,
                "mean": mean,
                "sem": sem,
                "trial_index": trial_index,
            }
            rows.append(new_row)

    results_data = pd.DataFrame(rows)
    data = Data(df=results_data)
    # Attach the data to the experiment without overwriting the previous data
    _ = experiment.attach_data(data=data, combine_with_last_data=True)

    # Marking the trial as completed so that it can be used for training the model
    trials = experiment.get_trials_by_indices(trial_indices=[trial_index])
    _ = trials[0].mark_completed()

    # Return the data that was attached
    return data, trial_index


def add_simulation_data_to_experiment(
    experiment: Experiment,
    parameters: list[str],
    simulation_inputs: Iterable[NDArray[np.float64]],
    simulation_outputs: Iterable[NDArray[np.float64]],
) -> tuple[Data, int]:
    """Add raw simulator data to an experiment.

    This function is used to add data from previously run simulations to an experiment.
    This is very useful if the simulator is slow or expensive to run and the data is already available.
    Using this function allows you to add the data to the experiment and then use it to train the model
    which can be used to collect new data by running new simulations.

    Args:
        experiment: The ax Experiment to add the data to.
        parameters: The names of the parameters used as input to the simulator.

            - This has to match the names of the parameters in the search space of the experiment.
            - This should match the order of the input dimensions in the simulation_inputs.

        simulation_inputs: The inputs to the simulator.

            - Iterable where each element is the parameters used in the simulation.
            - Each element should be a numpy array with shape (n_input_dims,)

        simulation_outputs: The outputs from the simulator.

            - Iterable where each element is multiple output from the simulation ran with the same input.
            - Each element should be a numpy array with shape (n_simulations_per_point,)
            - Only supports a single output dimension for now.

    Returns:
        - The data that was attached to the experiment.
        - The trial index of the trial that was created.
    """
    _: Any
    runner = experiment.runner
    assert isinstance(runner, LocalMetadataRunner), (
        f"Expected experiment.runner to be of type LocalMetadataRunner, got: {type(runner)}"
    )

    # Change format of inputs into parameterization dicts
    parameterizations: list[TParameterization] = []
    for x in simulation_inputs:
        flat_parameters = x.flatten()

        parameterizations.append({parameter_name: flat_parameters[i] for i, parameter_name in enumerate(parameters)})

    # Use the post processing from the evaluation function to get
    # the distribution parameters from the simulation outputs
    simulation_distribution_results = [
        runner.evaluation_function.post_process_simulation_output(y) for y in simulation_outputs
    ]

    # Fetch the mean and sem for each metric from the simulation distribution results
    distribution_metric_results: list[dict[str, float | tuple[float, float | None] | dict[str, float | None]]] = [
        {metric_name: distribution_result.metric_data(metric_name) for metric_name in experiment.metrics}
        for distribution_result in simulation_distribution_results
    ]

    # Use the helper function to add the metric data to the experiment
    return add_metric_data_to_experiment(experiment, parameterizations, distribution_metric_results)


def extract_data_from_experiment_as_json(
    experiment: Experiment,
) -> dict[int, dict[str, dict[str, dict[str, float | dict[str, float]]]]]:
    """Extracts the data from an ax Experiment.

    Args:
        experiment: The Ax experiment to extract the data from.

    Returns:
        A dictionary with the following structure:

        .. code-block:: json

            {
                "trial_index": {
                    "arm_name": {
                        "parameters": {
                            "parameter_name": "parameter_value"
                        },
                        "metrics": {
                            "metric_name": {
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

    trial_data: dict[int, dict[str, dict[str, dict[str, float | dict[str, float]]]]] = {}

    metric_data_by_arm = metric_data.groupby("arm_name")

    for trial in trials:
        if trial.index not in trial_data:
            trial_data[trial.index] = {}
        for arm in trial.arms:
            if arm.name not in trial_data[trial.index]:
                trial_data[trial.index][arm.name] = {
                    "parameters": {},
                    "metrics": {},
                }

            for parameter_name, parameter_value in arm.parameters.items():
                trial_data[trial.index][arm.name]["parameters"][parameter_name] = parameter_value

            arm_metric_data = metric_data_by_arm.get_group(arm.name)[["metric_name", "mean", "sem"]]
            for metric_name, mean, sem in arm_metric_data.to_numpy():
                trial_data[trial.index][arm.name]["metrics"][metric_name] = {"mean": mean, "sem": sem}

    return trial_data


def add_json_data_to_experiment(
    experiment: Experiment, json_data: dict[int, dict[str, dict[str, dict[str, float | dict[str, float]]]]]
) -> None:
    """Adds the data from a dictionary to an ax Experiment.

    Args:
        experiment: The ax Experiment to add the data to.
        json_data: The data to add to the ax Experiment. The structure should be the same as the output of
            extract_data_from_experiment.

    Example:
        .. code-block:: json

            {
                "trial_index": {
                    "arm_name": {
                        "parameters": {
                            "parameter_name": "parameter_value"
                        },
                        "metrics": {
                            "metric_name": {
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
        # The parameters should always be of type dict[str, float]
        # So we use cast to tell mypy that we know this is the case
        trial_parameters: list[dict[str, float]] = [
            cast(dict[str, float], arm_data["parameters"]) for arm_data in trial_arms.values()
        ]

        # The metrics should always be of type dict[str, dict[str, float]]
        # So we use cast to tell mypy that we know this is the case
        trial_metrics: list[dict[str, dict[str, float | None]]] = [
            cast(dict[str, dict[str, float | None]], arm_data["metrics"]) for arm_data in trial_arms.values()
        ]
        _ = add_metric_data_to_experiment(
            experiment=experiment,
            parameterizations=trial_parameters,
            metric_data=trial_metrics,
        )
