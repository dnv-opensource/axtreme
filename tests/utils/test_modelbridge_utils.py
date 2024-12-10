import numpy as np
from ax.core.observation import Observation, ObservationData, ObservationFeatures

from axtreme.utils.modelbridge_utils import observations_to_arrays


def test_observations_to_arrays():
    x = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([[10], [20], [30]])
    of = [ObservationFeatures(parameters={"x1": point[0], "x2": point[1]}) for point in x]
    od = [ObservationData(metric_names=["y"], means=point, covariance=np.eye(1)) for point in y]
    observations = [Observation(features=f, data=d) for f, d in zip(of, od, strict=False)]
    features, f, cov = observations_to_arrays(param_names=["x1", "x2"], outcomes=["y"], observations=observations)
    assert np.array_equal(features, x)
    assert np.array_equal(f, y)
    # There is only one output per x point. This will have perfect covariance with itself.
    assert np.array_equal(cov, np.ones([3, 1, 1]))
