"""Responsible for running simulators with a given set of Ax Parameters, and returning estimate with uncertainty."""

from dataclasses import dataclass

import numpy as np
from ax.core import TParameterization
from scipy.stats import rv_continuous

from axtreme.simulator import Simulator
from axtreme.utils import distibution_helpers


# QUESTION: is this the right place for this?
@dataclass
class SimulationPointResults:
    """For a point that has been simulation, stores the mean(s) and covariance(s).

    Args:
        means (NDArray[np.float64]): array of the mean paramter/metric estimates at a X point.
        cov (NDArray[np.float64]|None): covarianc matrix with uncertainty distibution of the metric estimates.
            - This can be None if the error is unknown
        parameter_names: list of names. Gives the index where relevant data is stored

    Design rational:
        - The primary purpose of this is to define the interface between the `Runner` which generate simulation results,
          and `Metric` which reports the required parts of the results for AX to then use.
        - The intent is to :
            - Explicitally define the interface information between the two (rather than use a dict)
            - Prevent `Runner` needing to know about specific structure of metric.
            - Prevent `Metric` needing to know the stucture of Runner
            - Esentially it mean the translatioin logic between these two components is contained in one discrete
              unit/object


    This is generated (in the `Runner`) when the simulation is evaluated for a specific `Trial`.
    It is attached to the `Trail.metadata`, and later read by `Metric.fetch_trial_data`.

    Args:
        - metric_name: The name of the metric that these results are relevant to
        - mean: The mean estimate of the metric for this particular trial
        - sem: Standard Error Measure, as defined `here <https://ax.dev/docs/glossary.html#sem>`_

    Background: For documenation on what can do into this object
        - `Ax tutorial on defining custom metrics <https://ax.dev/tutorials/gpei_hartmann_developer.html#8.-Defining-custom-metrics>`_
        - `Ax definition <https://ax.dev/docs/glossary.html#sem>`_
        - `For the possible things might want to contruct in the end <ax.core.data.Data.REQUIRED_COLUMNS>`_

    Todo:
        Revist if there is a better abstraction for this. Detail in Github issue #31.
    """

    metric_names: list[str]
    means: np.ndarray[tuple[int,], np.dtype[np.float64]]
    cov: np.ndarray[tuple[int, int], np.dtype[np.float64]] | None

    def metric_data(self, metric_name: str) -> dict[str, float | None]:
        """Construct the 'Metric data-related columns' as defined in ax.Data.

        This consists of:
            - "mean": mean estimate of this parameter
            - "sem": as defined `here <https://ax.dev/docs/glossary.html#sem>`_
        """
        if metric_name not in self.metric_names:
            msg = f"Metric name {metric_name} not in the results (metrics available for {self.metric_names})"
            raise KeyError(msg)

        idx = self.metric_names.index(metric_name)

        return {"mean": self.means[idx], "sem": np.sqrt(self.cov[idx][idx]) if self.cov is not None else None}


# TODO(sw): is there any meaningful difference between closure and partial here?
# TODO(ks): Make Protocol that is more general for EvaluationFunction and make the class below an instance of that


class EvaluationFunction:
    """Class for evaluating a simulation at a point, and fits a distribution to the results.

    Take ax.Parameter, runs the simulator at that point, and packages the result into the format later required
    for unpacking in the metrics.


    """

    def __init__(
        self,
        simulator: Simulator,
        output_dist: rv_continuous,
        parameter_order: list[str],
        n_simulations_per_point: int = 20,
    ) -> None:
        """Initializes the EvaluationFunction.

        Args:
            simulator: Conforming to the simulator in
            output_dist: Distribution that should be used to fit the response. This is the distribution of the noise
             in the simulation, at a given x,
            n_simulations_per_point: The number of simulations to run at each point.
            parameter_order: Order of features that the simulator expects.

                .. Note::
                    This is a temporary measure to check that the order isn't ever shuffled.

                .. Todo::
                    Got through ax and check is order is always respected, so we can remove this
        """
        self.simulator = simulator
        self.output_dist = output_dist
        self.parameter_order = parameter_order
        self.n_simulations_per_point = n_simulations_per_point

    def __call__(self, parameters: TParameterization) -> SimulationPointResults:
        """Gets the simulation results for an input point."""
        # Not sure if TParameterization always return a dict with parameters in the same order
        # This is small risk, but will have a big impact and be difficult to catch,
        # check it here until we are completely sure
        assert list(parameters.keys()) == self.parameter_order

        x = np.array([np.fromiter(parameters.values(), dtype=float)])

        y = self.run_simulator(x)

        # This only works for a single output dimension for now
        assert y.shape[0] == 1, f"simulation_result.shape[0] must be 1, got: {y.shape[0]}"

        y_flat = y.flatten()

        return self.post_process_simulation_output(y_flat)

    def run_simulator(
        self, x: np.ndarray[tuple[int, int], np.dtype[np.float64]]
    ) -> np.ndarray[tuple[int, int, int], np.dtype[np.float64]]:
        """Runs the simulator at a point and returns the results.

        Args:
            x: The points at which to run the simulator with shape (n_points, n_input_dims)

        Returns:
            The results of the simulator with shape (n_points, n_simulations_per_point, n_output_dims)

        Raises:
            AssertionError: If the shape of the output of self.simulator is not as expected.
        """
        y = self.simulator(x=x, n_simulations_per_point=self.n_simulations_per_point)

        assert y.ndim == 3, f"simulation_result.ndim must be 3, got: {y.ndim}"  # noqa: PLR2004
        assert (
            y.shape[1] == self.n_simulations_per_point
        ), f"simulation_result.shape[1] must be {self.n_simulations_per_point}, got: {y.shape[1]}"
        assert y.shape[2] == 1, f"simulation_result.shape[2] currenlty only support single output, got: {y.shape[2]}"

        return y

    def post_process_simulation_output(
        self, y: np.ndarray[tuple[int, int, int], np.dtype[np.float64]]
    ) -> SimulationPointResults:
        """Post process the simulation output be fitting a the distribution to the results.

        Args:
            y: The output of the simulation with shape (n_simulations_per_points,)

                .. Note::
                    This is the output of the simulator, at a single point and a single output dimension.

        Returns:
            A SimulationPointResults object with the mean and covariance of the fitted distribution.
        """
        # NOTE: the order/index matches that returned by rv_continous.fit
        # This can be explicity collected by using distribution_parameter_names_from_scipy(rv_continous)
        param_means, cov = distibution_helpers.fit_dist_with_uncertainty(y, self.output_dist)

        # TODO(sw): Will need to change when we look at correlated output becuase will need to look at the covariances
        # Will address this once we know what the correlated GP requires as input and output
        param_names = distibution_helpers.distribution_parameter_names_from_scipy(self.output_dist)

        return SimulationPointResults(metric_names=param_names, means=param_means, cov=cov)
