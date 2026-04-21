import torch
from scipy import stats
from scipy.stats import norm
from torch.distributions import Normal
from torch.utils.data import DataLoader, RandomSampler, TensorDataset

from axtreme.utils.brute_force import brute_force_calc


def test_brute_force_calc_matches_order_statistic_distribution():
    """Output samples should follow the distribution of the maximum of `period_length` i.i.d. draws.

    By ordered-statistics theory, if X_1, ..., X_n are i.i.d. with CDF F, the CDF of their maximum is F(x)^n.

    Setup: response is always Normal(0, 1) regardless of environment input, so each draw within a period is
    i.i.d. N(0, 1). With `period_length=100` draws per period, the CDF of the period maximum is:

        F_max(x) = F(x)^100

    Validated with a Kolmogorov-Smirnov test: the KS statistic measures the largest gap between the
    empirical CDF of `maxs` and the theoretical CDF. A p-value > 0.05 means we cannot reject that
    the samples come from the theoretical distribution.
    """
    _ = torch.manual_seed(0)

    period_length = 100
    num_estimates = 5_000

    # Env data content is irrelevant — response_params_func ignores it.
    dataset = TensorDataset(torch.zeros(1000, 2))
    dataloader = DataLoader(
        dataset,
        batch_size=period_length,
        sampler=RandomSampler(dataset, num_samples=period_length, replacement=True),
    )

    def response_params_func(env_batch: torch.Tensor) -> torch.Tensor:
        """Return Normal(0, 1) params for every input row."""
        n = env_batch.shape[0]
        return torch.stack([torch.zeros(n), torch.ones(n)], dim=-1)

    maxs, _ = brute_force_calc(
        dataloader=dataloader,
        response_params_func=response_params_func,
        response_dist_class=Normal,
        num_estimates=num_estimates,
    )

    # CDF of max(X_1,...,X_100) where X_i ~ N(0,1) is F(x)^100
    ks_result = stats.kstest(maxs.numpy(), lambda x: norm.cdf(x) ** period_length)
    assert ks_result.pvalue > 0.05, f"KS test failed: p={ks_result.pvalue:.4f}"


def test_brute_force_calc_matches_manual_calculation():
    """Output should match a simple hand-written torch reimplementation.

    Uses a near-zero response scale so ``sample() ≈ mean``, making results quasi-deterministic
    without needing to match RNG state between ``brute_force_calc`` and the manual loop.
    (The DataLoader iterator internally calls ``.random_()`` to generate a worker seed on each
    iteration, consuming RNG state that a plain Python loop does not replicate.)

    The response mean is set to the first column of env_data.  Row 1 (mean=3.0) is always the
    clear maximum, so the expected output is unambiguous across all estimates.
    """
    num_estimates = 5

    # First column = response mean per row.  Row 1 (3.0) is the dominant maximum.
    env_data = torch.tensor(
        [
            [1.0, 0.5],
            [3.0, 0.1],  # <- always the winner
            [2.0, 0.3],
        ]
    )
    dataset = TensorDataset(env_data)
    # Single batch, no shuffle → identical data every iteration.
    dataloader = DataLoader(dataset, batch_size=len(env_data), shuffle=False)

    scale_eps = 1e-6  # near-zero scale: sample() ≈ mean, ordering of rows is preserved

    def response_params_func(env_batch: torch.Tensor) -> torch.Tensor:
        means = env_batch[:, 0]
        scales = torch.full((env_batch.shape[0],), scale_eps)
        return torch.stack([means, scales], dim=-1)

    maxs, maxs_location = brute_force_calc(
        dataloader=dataloader,
        response_params_func=response_params_func,
        response_dist_class=Normal,
        num_estimates=num_estimates,
    )

    # --- Hand-calculated expected values ---
    # With scale ≈ 0, sample() ≈ mean = env_batch[:, 0].  Data is identical every period,
    # so the same row wins every estimate.
    expected_maxs = torch.tensor([3.0] * num_estimates)
    expected_locations = torch.tensor([[3.0, 0.1]] * num_estimates)

    torch.testing.assert_close(maxs, expected_maxs, atol=scale_eps * 10, rtol=0.0)
    torch.testing.assert_close(maxs_location, expected_locations)
