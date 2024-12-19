"""Helper for finding inverse cdf for function that do not have an icdf method."""

from itertools import product

import torch
from scipy.optimize import RootResults, root_scalar
from torch.distributions import Distribution

from axtreme.distributions.helpers import dist_cdf_resolution
from axtreme.distributions.utils import dist_dtype, index_batch_dist

# pyright: reportUnnecessaryTypeIgnoreComment=false


def icdf(
    dist: Distribution,
    q: torch.Tensor,
    max_acceptable_error: float,
    opt_bounds: torch.Tensor,
) -> torch.Tensor:
    """Calculated the inverse CDF for each distribution in a batch.

    This method is useful when the distrbution does not have an icdf method. In this case the CDF can be run repeatidly
    until we find the `x` where `cdf(`x`) == quantile`. We use optimisation to make this search effecient.

    Args:
        dist: (`*batch_shape`,) mixture distribution producing events of event_shape samples
        q: quantile to find the inverse cdf of. Must be boardcastable up to (`*batch_shape`,). Must not have more
           dimensions than `*batch_shape` (only 1 q can be passed to each of the distributions in the batch.)
        opt_bounds:(2,`*batch`). The lower and lower x bounds withing which the q can be found. If unknown, set to
            a very large range. `opt_bounds.shape[1:]` must be broadcastable to `dist.batch_shape`.
        max_acceptable_error: return values must be within `icdf(q +/- max_acceptable_error)`

    Returns:
        Tensor of shape (dist.batch_shape) of the icdf(x) results.
    """
    # Expand insure q.dim() < len(dist.batch_shape)
    q_expanded = q.expand(dist.batch_shape)
    bounds_expanded = opt_bounds.expand(torch.Size([2]) + dist.batch_shape)

    idx_ranges = [range(dim) for dim in dist.batch_shape]
    # List of all the indexes in the batch shape, in the same order as tensor.flatten()
    idx_list = product(*idx_ranges)

    # order of items matches flatten.
    results = []
    for idx in idx_list:
        single_dist = index_batch_dist(dist, idx)

        bounds = bounds_expanded[:, *idx]  # type: ignore  # noqa: PGH003 mypy doesn't recognise :

        result = icdf_1d(
            dist=single_dist,
            quantile=q_expanded[idx].item(),
            max_acceptable_error=max_acceptable_error,
            bounds=tuple(bounds.tolist()),
        )
        results.append(result)

    return torch.tensor(results).reshape(dist.batch_shape)


def icdf_1d(
    dist: Distribution,
    quantile: float,
    max_acceptable_error: float,
    bounds: tuple[float, float],
) -> torch.Tensor:
    """Calculates inverse CDF values for distributions which do not have a icdf function.

    Some distributions do not have an inverse CDF function. In this case the CDF can be run repeatidly until we find the
    `x` where `cdf(`x`) == quantile`. We use optimisation to make this search effecient. This function only supports
    distribution with 1d input and output.

    Args:
        dist: This expects a Distribution like object. The specific methods and attributes required are:

            - methods: `cdf()`, and potentially `log_prob()` depending on the optimiser used.
            - Attributes: None.

        quantile: The quantile for which to find the corresponding x value.
        max_acceptable_error: return values must be within `icdf(q +/- max_acceptable_error)`
        bounds: The bounds to serach for the root in

    Returns:
        A tensor of shape (1,) of the x value corresponding to the quantile.

    Details:
        There are a number of way numerical issues can effect this results:
            - Case 1: Function output (dist.cdf) is not precise enough to support max_acceptable_error.
            - Case 2: Function internals are not precise enough to support max_acceptable_error (e.g they introduce
              numeric error).
            - Case 3: Optimiation input (x), is not accurate enough to support step sizes need to achieve
              max_acceptable_error.
            - Case 4: The optimisation process does not find a suitable result.
            - Case 5: The output datatype truncation is not precise enough, failing max_acceptable_error after
              truncation.
    """
    dtype = dist_dtype(dist)

    # Checks case 1 and 2.
    if dist_cdf_resolution(dist) > max_acceptable_error:
        msg = (
            "The distribution provided does not have suitable resolution (numeric precision) for the"
            f" max_accptable_error specified. {dist_cdf_resolution(dist) = } and {max_acceptable_error = }"
            "increase the max_acceptable_error or use bigger dtype in the distribution."
        )
        raise TypeError(msg)

    # Scipy operates in float64. If scipy inputs are clipped to float32 before the function is run it can cause issues.
    def f(x: float) -> float:
        # TODO(sw 2024-12-17): Does converting to float64 mean the cdf automatically runs in float64? Is the reasonable
        # if users specifically request a float32 is used?.
        x_tensor = torch.tensor([x], dtype=torch.float64)
        return (dist.cdf(x_tensor) - quantile).item()

    # Scipy doesn't have an option to set tolerance in terms of the output, only in terms of x. Output tolerance is the
    # termination criteria we care about, so we need to check this manually.
    for maxiter in [30, 100, 300]:
        # NOTE: better documentation of bisect args can be found scipy.optimize._zeros_
        # NOTE: root_scalar overides the following arguments {disp = False, full_output = True}
        opt_result: RootResults = root_scalar(
            f,
            bracket=bounds,
            method="bisect",
            maxiter=maxiter,
        )

        # NOTE: "output resolution check" gaurentees the distribution has enough accuracy to produce the required output
        # here we use the same dtype as scipy operates in, as we don't want to truncate the input. The intent here is to
        # see if a good enough solution has been found. If there are tunction errors with the chosen data type, that
        # should be raised later.
        x64 = torch.tensor(opt_result.root, dtype=torch.float64)
        q_error_opt = dist.cdf(x64) - quantile

        if q_error_opt.abs() < max_acceptable_error:
            # Termination criteria has been met.
            break

    # Check case 3:
    # If converged, means we have hit the minimum x step size available for that number
    # NOTE: Scipy automatically terminates when the minimum numeric precision is hit for an x value. This scales with
    # the magnitude of x.
    if (q_error_opt.abs() > max_acceptable_error) and opt_result.converged:
        msg = (
            "Optimisation has reached minimum step size in x, but the error is still greater than max_acceptable_error."
            " The datatype used here is python/numpy float - increasing these beyond float 64 is not currently"
            " supported. Increase max_acceptable_error."
        )
        raise RuntimeError(msg)

    # Check case 4:
    # NOTE: Due to case 1 and 2 gauranteeing the output is accurate enough, this is suffecient to check optimiation
    # result.
    if q_error_opt.abs() > max_acceptable_error:
        msg = (
            f"Optimisation result has error  ({q_error_opt})  greater than  {max_acceptable_error=}."
            " Is may be cause by insuffecient optimsation iterations"
        )
        raise RuntimeError(msg)

    # Check case 5:
    x = torch.tensor(opt_result.root, dtype=dtype)
    q_error = dist.cdf(x) - quantile
    if (q_error_opt.abs() < max_acceptable_error) and (q_error.abs() > max_acceptable_error):
        msg = (
            f"Succesful optimisation results fail once trucated to the specifed dtype ({dtype =})."
            " Increase max_acceptable_error or dist.dtype."
        )
        raise TypeError(msg)

    return x
