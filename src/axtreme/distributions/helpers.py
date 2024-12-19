"""Helpers built ontop of the distributions module.

Todo:
    These are seperate from `utils` becuase of circular import caused by `ApproximateMixture`. Potentially there is a
    more elegant solution.
"""

import torch
from torch.distributions import Distribution, MixtureSameFamily

from axtreme.distributions.mixture import ApproximateMixture
from axtreme.distributions.utils import dist_dtype


def dist_cdf_resolution(dist: Distribution) -> float:
    """To what resolution (number of decimal places) is cdf output accurate to.

    This is effected by:

        - The numeric precision of the datatype q is stored with (e.g float32)
        - The internal calculation and numerical error incurrred by them.

    Return:
        10**precision, where precision is the decimal position. E.g if accurate to 3 decimal places, return 0.001

    Note:
        - We are interested in the `cdf(x) -> q` accuracy, as this is used heavily.
        - `icdf(q) -> x` does not have the same resolution as detailed here. The difference in results increases with x.
        - This number/method is determined emprically. Tests show it is a good bound for float16 and float32.
            Assumed to hold for float64
    """
    if isinstance(dist, ApproximateMixture):
        return approx_mixture_cdf_resolution(dist)
    if isinstance(dist, MixtureSameFamily):
        return mixture_cdf_resolution(dist)

    dtype = dist_dtype(dist)
    # NOTE: The numeric precision to which q can be represented also appears to bound the cdf output.
    # This has been determined emperically. See unit tests for details.
    # NOTE: this error is larger than just the rounding error we would expect for q (eps/2)
    return torch.finfo(dtype).resolution


def mixture_cdf_resolution(dist: MixtureSameFamily) -> float:
    """To what resolution (number of decimal places) is cdf output accurate to.

    This compared to `dist_cdf_resolution` this is also impacted by `weights*components`.

    This is effected by:

        - The numeric precision of the datatype q is stored with (e.g float32)
        - The internal calculation and numerical error incurred by them.
        - Combination of `weight*components`

    Return:
        10**precision, where precision is the decimal position. E.g if accurate to 3 decimal places, return 0.001

    Note:
        - We are interested in the `cdf(x) -> q` accuracy, as this is used heavily.
        - `icdf(q) -> x` does not have the same resolution as this. The difference in results increases with x.
        - This number/method is determined emprically. Tests show it is a good bound for float16 and float32.
          Assumed to hold for float64
    """
    dtype = dist_dtype(dist)
    # NOTE: The numeric precision to which q can be represented also appears to bound the cdf output.
    # This has been determined emperically. See unit tests for details. As per the unit test, we add a safety factor of
    # 10, as we are not certain the unittest represents worst case behaviour.
    # NOTE: this error is larger than just the rounding error we would expect for q (eps/2)
    return torch.finfo(dtype).resolution * 10


def approx_mixture_cdf_resolution(dist: ApproximateMixture) -> float:
    """To what resolution (number of decimal places) is cdf output accurate to.

    This is identical to `mixture_cdf_resolution` except for the conservatism introduced by the approximation.
    See `ApproximateMixture` "Impact of Approximation:" for details
    """
    dtype = dist_dtype(dist)
    resolution = mixture_cdf_resolution(dist)
    # As per ApproximateMixture the approximation error is bounded by finfo.eps
    return max(resolution, torch.finfo(dtype).eps)
