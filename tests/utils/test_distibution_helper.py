from typing import Any

import numpy as np
import pytest
from scipy import stats
from scipy.stats._distn_infrastructure import rv_continuous

from axtreme.utils import distibution_helpers


@pytest.mark.parametrize(
    "dist, expected",
    [
        (stats.gumbel_r, ["loc", "scale"]),
        (stats.weibull_max, ["c", "loc", "scale"]),
        (stats.beta, ["a", "b", "loc", "scale"]),
    ],
)
def test_distribution_parameter_names_from_scipy(dist: rv_continuous, expected: list[str]):
    result = distibution_helpers.distribution_parameter_names_from_scipy(dist)
    assert result == expected


# TODO(sw): How do we chance that the uncertainty (cov) we are getting is reasonable?
@pytest.mark.parametrize(
    "dist, params",
    [
        pytest.param(stats.gumbel_r, {"loc": 4, "scale": 1}, id="gumbel_r"),
        pytest.param(
            stats.weibull_max,
            {"c": 1, "loc": 4, "scale": 1},
            marks=pytest.mark.xfail(reason="Currently only support gumbel"),
            id="weibull_max",
        ),
    ],
)
def test_fit_dist_with_uncertainty(dist: rv_continuous, params: dict[str, Any]):
    sample = dist.rvs(**params, size=10_000, random_state=7)
    means, _ = distibution_helpers.fit_dist_with_uncertainty(sample, dist)
    np.testing.assert_allclose(means, list(params.values()), atol=1e-2)
