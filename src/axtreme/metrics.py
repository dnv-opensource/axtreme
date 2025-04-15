"""Additional Metric implemenations."""

import copy
from typing import TYPE_CHECKING, Any

import pandas as pd
from ax import Arm, BatchTrial, Data, Metric, Models, Trial
from ax.core.base_trial import BaseTrial
from ax.core.metric import MetricFetchResult
from ax.utils.common.result import Ok

from axtreme import qoi
from axtreme.qoi.qoi_estimator import QoIEstimator

if TYPE_CHECKING:
    from axtreme.evaluation import SimulationPointResults
from typing import cast


class LocalMetadataMetric(Metric):
    """This metric retrieves its results form the trial metadata.

    The simple example run the simultion within this function call
    (e.g. `see <https://ax.dev/tutorials/gpei_hartmann_developer.html#8.-Defining-custom-metrics>`_)
    In general, this method should only 'fetch' the results from somewhere else where they have been run.
    For example, Runner deploys simulation of remote, this connects to remote and collects result.
    This is local implementation of this patter, where results are stored on trail metadata.

    This is useful when:
    - Running a single simulation produces multiple output metrics
    (meaning the simulation doesn't need to be run as many times)
    - Want to execute the simulation when `trail.run()` is called

    Note:
        This object is coupled with LocalMetadataRunner, through SimulationPointResults

    Background:

    Flow:
    - trial.run() called, internally call the runner, and puts the resulting data into metadata
    - Later Metric.fetch_trial_data(trial) is called Therefore, Metric has access to all the "bookkeeping"
    trial information directly, the only thing that should be in metadata is run result.
    """

    def fetch_trial_data(
        self,
        trial: BaseTrial,
        **kwargs: Any,  # noqa: ANN401, ARG002 NOTE: kwargs is needed to match the signature of the parent class.
    ) -> MetricFetchResult:
        """Fetches the data from the trial metadata."""
        arm = _single_arm_trail(trial)

        point_result: SimulationPointResults = trial.run_metadata["simulation_result"]
        metrics_columns = point_result.metric_data(self.name)
        # The data that must be contained in the results can be found here: Data.REQUIRED_COLUMNS
        # Required keys are {"arm_name","metric_name", "mean", "sem"}
        # Additional info can be found here: ax.core.data.BaseData
        data = {
            "arm_name": arm.name,
            "trial_index": trial.index,
            "metric_name": self.name,
            **metrics_columns,
        }

        return Ok(value=Data(df=pd.DataFrame([data])))


def _single_arm_trail(trial: BaseTrial) -> Arm:
    """Ensure the trial is a Trial with a single arm.

    This helper is useful as Metric subclasses need to support BaseTrials, otherwise they break polymorphism
    (e.g SubClassMetric can no longer be used in place of it parent class Metric). In Ax there are 2 type of trials
    (Trial and BatchTrial, see https://ax.dev/docs/core/#trial-vs-batch-trial), and we are typically only interested in
    supporting `Trail`. This helper deals with the checking and typing.

    Args:
        trial: The trial to check.

    Return:
        The arm of a single arm trail, or exception.
    """
    if isinstance(trial, BatchTrial):
        raise NotImplementedError("BatchTrial is not supported for LocalMetadataMetric.")

    # We proceed with duck typing rather than explicitly checking type Trial
    trial = cast("Trial", trial)
    # This internally throws an error if more than 1 arm is attached.
    arm = trial.arm

    if arm is None:
        raise ValueError("Trial has no Arm. It cannot be evaluated.")

    return arm


class QoIMetric(Metric):
    """Helper for recording the QoI estimate over the course of an Experiment.

    This helper records the score of a QoIEstimator. Internally it trains a GP on all non-tracking metrics for trials up
    to the current trial (e.g if the `Experiment` has 10 trails total, the QoI estimate for trail 5 will use all trials
    0-5 inclusive in the GP). The QoIEstimator is then run using the GP.

    This metric should be used as a tracking metric (e.g `_tracking_metrics` attribute `Experiment`).
    `_tracking_metrics` signify that these metrics should not be modelled with a GP (Note: this still need to be
    explicitly set when creating GPs).

    Pseudo code Example:
    >>> qoi_estimator = qoi.GPBruteForce(...)  # or any other QoIEstimator
    >>> qoi_metric = QoIMetric(name="QoIMetric", qoi_estimator=qoi_estimator, attach_transforms=True)
    >>> experiment = Experiment(...)
    >>> experiment.add_tracking_metric(qoi_metric)
    """

    # What is the metric

    # Note on subclasses: input signature can wbe whatever you want
    def __init__(
        self,
        name: str,
        qoi_estimator: QoIEstimator,
        minimum_data_points: int = 5,
        lower_is_better: bool | None = None,
        properties: dict[str, Any] | None = None,
        *,
        attach_transforms: bool = False,
    ) -> None:
        """Initialize the QoIMetric.

        Args:
            name: The name of the metric.
            qoi_estimator: The QoI estimator to use to calculate the metric.
            minimum_data_points: The minimum number of datapoints the experiment must have before the GP is trained and
                the QoI is actually run.
            lower_is_better: Flag for metrics which should be minimized. Typically we are not interested in
                minimising/maximising QoIs so this values should be `None`.
            properties: Dictionary of this metric's properties.
            attach_transforms: If True, attaches the input and outcome transforms required of the GP to operate in the
                problem space. This assume `qoi_estimator` have attributes `input_transform` and `outcome_transform`
                where these should be attached. TODO(sw 11-4-25): remove when issue #19 is addressed.
        """
        super().__init__(name=name, lower_is_better=lower_is_better, properties=properties)

        self.qoi_estimator = copy.deepcopy(qoi_estimator)
        self.minimum_data_points = minimum_data_points
        self.attach_transforms = attach_transforms

    def fetch_trial_data(
        self,
        trial: BaseTrial,
        **kwargs: Any,  # noqa: ANN401, ARG002 NOTE: kwargs is needed to match the signature of the parent class.
    ) -> MetricFetchResult:
        """Fetch the data for a trial.

        See class docstring for overview.

        Returns:
            The data result in effect byt the amount of data available by this trial. If v
            - available data < `minimum_data-points`: `mean` and `sem` are NaN.
            - available data >= `minimum_data-points`: `mean` is the mean QoIEstimate, `sem` is the standard error of
                the measure deviation. The standard error corresponds to the standard deviation of the distribution as
                each sample is a prediction of the measure (the QoI).
        """
        arm = _single_arm_trail(trial)
        exp = trial.experiment

        qoi_mean = float("nan")
        qoi_sem = float("nan")

        # Trail has 0 based indexing, add 1 to get count of data points
        if (trial.index + 1) >= self.minimum_data_points:
            non_tracking_metrics = exp.optimization_config.metrics
            # Its possible the experiment has more trails than the current one.
            # The calculation at this point should only be with the data seen prior and including the current trial.
            data = exp.fetch_trials_data(
                trial_indices=range(trial.index + 1),
                # Tracking metric need to be excluded to avoid recursion
                metrics=list(non_tracking_metrics.values()),
            )

            botorch_model_bridge = Models.BOTORCH_MODULAR(experiment=exp, data=data, fit_tracking_metrics=False)

            # Likely removed as part of issue #19
            if self.attach_transforms:
                self.qoi_estimator = qoi.utils.attach_transforms_to_qoi_estimator(
                    botorch_model_bridge, self.qoi_estimator
                )

            model = botorch_model_bridge.model.surrogate.model
            estimates = self.qoi_estimator(model)
            qoi_mean = float(self.qoi_estimator.mean(estimates))
            qoi_sem = float(self.qoi_estimator.var(estimates) ** 0.5)

        # The data that must be contained in the results can be found here: Data.REQUIRED_COLUMNS
        # Required keys are {"arm_name","metric_name", "mean", "sem"}
        # Additional info can be found here: ax.core.data.BaseData

        # TODO(sw 11/04/25): can we attach the raw values qoi predictions to this as well?
        # They attach additional stuff here: .venv\Lib\site-packages\ax\metrics\noisy_function.py
        # Suspect it works like a standard pd dataframe - can add extra stuff in if you would like to
        data = {
            "arm_name": arm.name,
            "trial_index": trial.index,
            "metric_name": self.name,
            "mean": qoi_mean,
            "sem": qoi_sem,
        }

        return Ok(value=Data(df=pd.DataFrame([data])))
