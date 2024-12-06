"""Additional Metric implemenations."""

from typing import TYPE_CHECKING, Any

import pandas as pd
from ax import Data, Metric
from ax.core.base_trial import BaseTrial
from ax.core.metric import MetricFetchResult
from ax.utils.common.result import Ok

if TYPE_CHECKING:
    from axtreme.evaluation import SimulationPointResults


class LocalMetadataMetric(Metric):
    """This metric retrieves its results form the trial metadata.

    The simple example run the simultion within this function call (e.g. `see <https://ax.dev/tutorials/gpei_hartmann_developer.html#8.-Defining-custom-metrics>`_)
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
    commi
        - trial.run() called, internally call the runner, and puts the resulting data into metadata
        - Later Metric.fetch_trial_data(trial) is called Therefore, Metric has access to all the "bookkeeping"
          trial information directly, the only thing that should be in metadata is run result.
    """

    # TODO(sw): Make it explictly support single arm trial
    def fetch_trial_data(  # noqa: D102
        self,
        trial: BaseTrial,
        **kwargs: Any,  # noqa: ANN401, ARG002 NOTE: kwargs is needed to match the signature of the parent class.
    ) -> MetricFetchResult:
        records = []
        for arm_name in trial.arms_by_name:
            point_result: SimulationPointResults = trial.run_metadata["simulation_result"]
            metrics_columns = point_result.metric_data(self.name)
            # The data that must be contained in the results can be found here: Data.REQUIRED_COLUMNS
            # Required keys are {"arm_name","metric_name", "mean", "sem"}
            # Additional info can be found here: ax.core.data.BaseData
            records.append(
                {
                    "arm_name": arm_name,
                    "trial_index": trial.index,
                    "metric_name": self.name,
                    **metrics_columns,
                }
            )

        return Ok(value=Data(df=pd.DataFrame.from_records(records)))
