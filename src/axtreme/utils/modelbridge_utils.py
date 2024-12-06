"""This builds of the utils provided in ax.modelbridge.modelbridge_utils."""

import numpy as np
from ax.core.observation import Observation
from ax.modelbridge import modelbridge_utils
from numpy.typing import NDArray


def observations_to_arrays(
    param_names: list[str], outcomes: list[str], observations: list[Observation]
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Helper method to convert Ax results (Observation) to Numpy arrays of a standard format.

    This is useful e.g. when converting the output of `ax.model_bridge.base.ModelBridge.get_training_data` into "x" and
    "y" and "covariance" arrays.

    Args:
        param_names: List of the parameter names to report, and in what order.
        outcomes: list of the metric/target names to report, in this order.
            TODO(sw): work out if the order ever changes. If not, we should remove this.
        observations: The list of observations to convert.

    Returns:
        - features: An (n_observation, n_features) array of features.
        - f: An (n_observation, m_metrics) array of prediction means.
        - cov: An (n_observation, m_metrics, m_metrics) array of covariance between metric per x point.

    Example:
        >>> import numpy as np
        >>> from ax.core.observation import Observation, ObservationData, ObservationFeatures
        >>> X = np.array([[1, 2], [3, 4], [5, 6]])
        >>> Y = np.array([[10], [20], [30]])
        >>> of = [ObservationFeatures(parameters={"x1": x[0], "x2": x[1]}) for x in X]
        >>> od = [ObservationData(metric_names=["y"], means=y, covariance=np.eye(1)) for y in Y]
        >>> observations = [Observation(features=f, data=d) for f, d in zip(of, od)]
        >>> features, f, cov = observations_to_arrays(param_names=["x1", "x2"], outcomes=["y"], observations=observations)
        >>> assert np.array_equal(features, X)
        >>> assert np.array_equal(f, Y)
        >>> # There is only one output per x point. This will have perfect covariance with itself.
        >>> assert np.array_equal(cov, np.ones([3, 1, 1]))
    """  # noqa: E501
    features_array = modelbridge_utils.observation_features_to_array(param_names, [o.features for o in observations])
    f, cov = modelbridge_utils.observation_data_to_array(outcomes, [o.data for o in observations])
    return features_array, f, cov
