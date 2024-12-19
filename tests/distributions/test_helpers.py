import pytest
import torch
from torch.distributions import Categorical, Distribution, Gumbel, LogNormal, MixtureSameFamily

from axtreme.distributions.helpers import dist_cdf_resolution, mixture_cdf_resolution

# pyright: reportUnnecessaryTypeIgnoreComment=false


# TODO(sw 2024-12-14): Test more distributions
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("dist_class", [Gumbel, LogNormal])
def test_dist_cdf_resolution(dtype: torch.dtype, dist_class: type[Distribution]):
    """Bounds the numeric error of a dtype.

    A higher resolution dtype is considered to be the ground truth.
    This is a bruteforce appraoch that checks a large number of calculations.
    """
    # Set up the problem
    loc = torch.tensor([0.0], dtype=dtype)
    scale = torch.ones_like(loc, dtype=dtype)
    dist: Distribution = dist_class(loc, scale)  # type: ignore  # noqa: PGH003
    dist_ground_truth = dist_class(loc.type(torch.float64), scale.type(torch.float64))  # type: ignore  # noqa: PGH003
    finfo = torch.finfo(dtype)

    # See ApproximateMixture for why these quantiles are used as bounds
    q_lower = torch.tensor(finfo.eps, dtype=dtype)
    q_upper = torch.tensor(1 - finfo.eps, dtype=dtype)
    x_lower = dist.icdf(q_lower)
    x_upper = dist.icdf(q_upper)
    # Linspace produces nans if asked to create more possitions than the dtype supports, this heuristic avoids this
    n_points = int(min(100_000_000, 2 * finfo.max))
    x = torch.linspace(x_lower.item(), x_upper.item(), n_points, dtype=dtype)
    q = dist.cdf(x)
    q_ground_truth = dist_ground_truth.cdf(x)

    assert ((q - q_ground_truth).abs() < dist_cdf_resolution(dist)).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_mixture_dist_cdf_resolution(dtype: torch.dtype):
    """Bounds the numeric error of a dtype.

    A higher resolution dtype is considered to be the ground truth.
    This is a bruteforce appraoch that checks a large number of calculations.
    """
    # Set up the problem
    n_posterior_samples = 2
    n_env_samples_per_period = 5

    # fmt: off
    loc = torch.zeros(n_posterior_samples, n_env_samples_per_period)*0.0
    scale = torch.ones_like(loc, dtype=dtype)
    mix = Categorical(torch.ones(n_env_samples_per_period, dtype=dtype))
    # fmt: on
    comp = Gumbel(loc=loc.type(dtype), scale=scale.type(dtype))
    dist = MixtureSameFamily(mix, comp)

    mix_ground_truth = Categorical(torch.ones(n_env_samples_per_period, dtype=torch.float64))
    comp_ground_truth = Gumbel(loc=loc.type(dtype), scale=scale.type(torch.float64))
    dist_ground_truth = MixtureSameFamily(mix_ground_truth, comp_ground_truth)

    finfo = torch.finfo(dtype)

    # See ApproximateMixture for why these quantiles are used as bounds
    x_lower = comp.icdf(finfo.eps).min()
    x_upper = comp.icdf(1 - finfo.eps).max()

    # Linspace produces nans if asked to create more possitions than the dtype supports, this heuristic avoids this
    n_points = int(min(10_000_000, 2 * finfo.max))
    x = torch.linspace(x_lower.item(), x_upper.item(), n_points, dtype=dtype)
    # unsqueeze to broadcast across the batched
    x = x.unsqueeze(-1)
    q = dist.cdf(x)
    q_ground_truth = dist_ground_truth.cdf(x)

    assert ((q - q_ground_truth).abs() < mixture_cdf_resolution(dist)).all()
