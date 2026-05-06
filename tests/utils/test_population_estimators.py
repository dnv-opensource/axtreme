# sourcery skip: no-loop-in-tests

# %%
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from scipy.stats import binom
from scipy.stats import t as scipy_student_t
from torch import Tensor, tensor
from torch.distributions import Distribution, Gumbel, StudentT, Uniform, Weibull

from axtreme.utils.population_estimators import (
    estimate_pdf_value_from_sample,
    sample_mean_se,
    sample_median_se,
    sample_quantile_estimate,
)


@pytest.mark.non_deterministic
@pytest.mark.parametrize(
    "true_dist",
    [
        # Try a variety of different distributions
        (Weibull(3, 2)),
        (Uniform(3, 5)),
        (Gumbel(3, 2)),
        # See the impact of larger mean
        (Gumbel(30, 2)),
        # see impact of larger variance:
        (Gumbel(30, 6)),
    ],
)
def test_sample_mean_se(true_dist: Distribution):
    """Checks if the distribution returned by sample_mean_se is reasonable.

    The confidence bounds returned by this method are frequently used. We want to check that they are well calibrated.
    If we obtain at 95% confidence from a sample, 95% of the time the true mean should fall within these bounds. Higher
    or lower percentage represent miscallibrate.

    NOTE: sample_mean_se is primarily used as a diagnostic, so we are willing to accept some error in the results.

    Args:
        true_dist: The true distribution from which to generate samples from. The method has no restriction of the type
          of sample distribution.
    """
    n_samples_per_est = 20
    n_ests = 5000

    true_mean = true_dist.mean

    samples = true_dist.sample(torch.Size([n_ests, n_samples_per_est]))

    # If the confidence bounds are good, the true mean should fall in the range [0.025, 0.975] 95% of the time.
    # Therefore in our results, we should see q(true_mean) exceeding 0.975 or below 0.025 5% percent of the time.
    quantil_of_true_mean = []
    for sample in samples:
        dist: StudentT = sample_mean_se(sample)
        # The below raises a NotImplementedError, a bit a a hacky fix to switch scipy dist.
        # confidence_region = dist.cdf(1)
        dist_scipy = scipy_student_t(df=int(dist.df), loc=float(dist.loc), scale=float(dist.scale))
        quantil_of_true_mean.append(dist_scipy.cdf(true_mean))

    results = torch.tensor(quantil_of_true_mean)
    percent_outside_of_bounds = torch.logical_or(results > 0.975, results < 0.025).float().mean()

    assert float(percent_outside_of_bounds) == pytest.approx(0.05, abs=0.03)


@pytest.mark.non_deterministic
@pytest.mark.parametrize(
    "true_dist",
    [
        # Try a variety of different distributions
        (Weibull(3, 2)),
        (Uniform(3, 5)),
        (Gumbel(3, 2)),
        # See the impact of larger meanS
        (Gumbel(30, 2)),
        # see impact of larger variance:
        (Gumbel(30, 6)),
    ],
)
def test_sample_median_se(true_dist: Distribution):
    """Checks if the distribution returned by sample_median_se is reasonable.

    The confidence bounds returned by this method are frequently used. We want to check that they are well calibrated.
    If we obtain at 95% confidence from a sample, 95% of the time the true mean should fall within these bounds. Higher
    or lower percentage represent miscallibrate.

    NOTE: sample_median_se is primarily used as a diagnostic, so we are willing to accept some error in the results.

    Args:
        true_dist: The true distribution from which to generate samples from. The method has no restriction of the type
          of sample distribution.
    """
    n_samples_per_est = 50
    n_ests = 5000
    true_median = true_dist.icdf(tensor(0.5))

    samples: Tensor = true_dist.sample(torch.Size([n_ests, n_samples_per_est]))

    # If the confidence bounds are good, the true median should fall in the range [0.025, 0.975] 95% of the time.
    # Therefore in our results, we should see q(true_median) exceeding 0.975 or below 0.025 5% percent of the time.
    quantile_of_true_median = []
    for sample in samples:
        dist = sample_median_se(sample)
        quantile_of_true_median.append(dist.cdf(true_median))

    results = torch.tensor(quantile_of_true_median)
    percent_outside_of_bounds = torch.logical_or(results > 0.975, results < 0.025).float().mean()

    assert float(percent_outside_of_bounds) == pytest.approx(0.05, abs=0.05)


@pytest.mark.non_deterministic
@pytest.mark.parametrize(
    "true_dist",
    [
        # Try a variety of different distributions
        (Weibull(3, 2)),
        (Uniform(3, 5)),
        (Gumbel(3, 2)),
        # See the impact of larger meanS
        (Gumbel(30, 2)),
        # see impact of larger variance:
        (Gumbel(30, 6)),
    ],
)
def test_estimate_pdf_value_from_sample_at_median(true_dist: Distribution):
    """This tests for degradation in the performance on estimate_pdf_value_from_sample_at_median.

    The bounds required depend on `n_samples_per_est`. `visualise_performance_of_estimate_pdf_value_from_sample` can be
    used to determine them.
    """
    n_ests = 5000
    n_samples_per_est = 50

    q: float = 0.5
    _x: Tensor = true_dist.icdf(tensor(q))
    true_pdf_value = true_dist.log_prob(_x).exp()
    x: float = float(_x)

    samples = true_dist.sample(torch.Size([n_ests, n_samples_per_est]))

    pdf_ests = [estimate_pdf_value_from_sample(sample, x) for sample in samples]

    results = torch.tensor(pdf_ests)

    assert float(results.mean()) == pytest.approx(true_pdf_value, rel=0.2)
    coefficient_of_variation = results.std() / results.mean()
    assert coefficient_of_variation < 0.2


def visualise_performance_of_estimate_pdf_value_from_sample():
    """Show the bias and variance of `estimate_pdf_value_from_sample` an n_samples increases"""

    def test_estimate_pdf_value_from_sample(true_dist: Distribution, n_samples_per_est: int = 50, n_ests: int = 5000):
        q: float = 0.5
        _x: Tensor = true_dist.icdf(tensor(q))
        true_pdf_value = true_dist.log_prob(_x).exp()
        x: float = float(_x)

        samples = true_dist.sample(torch.Size([n_ests, n_samples_per_est]))

        # If the confidence bounds are good, the true mean should fall in the range [0.025, 0.975] 95% of the time.
        # Therefore in our results, we should see q(true_mean) exceeding 0.975 or below 0.025 5% percent of the time.
        pdf_ests = [estimate_pdf_value_from_sample(sample, x) for sample in samples]

        results = torch.tensor(pdf_ests)

        # assert float(results.mean()) == pytest.approx(true_pdf_value, rel=0.01)
        coeffeceint_of_variation = results.std() / results.mean()
        # assert coeffeceint_of_variation < 0.01

        return true_pdf_value, results.mean(), coeffeceint_of_variation

    dists = {
        "Weibull(3, 2)": (Weibull(3, 2)),
        "Uniform(3, 5)": (Uniform(3, 5)),
        "Gumbel(3, 2)": (Gumbel(3, 2)),
        # See the impact of larger meanS
        "Gumbel(30, 2)": (Gumbel(30, 2)),
        # see impact of larger variance:
        "Gumbel(30, 6)": (Gumbel(30, 6)),
    }
    n_samples_list = [11, 22, 44, 88, 176]

    # Placeholder for results
    results: dict[str, dict[str, list[float]]] = {
        name: {"n_samples": [], "true_pdf_value": [], "mean_est": [], "cov": []} for name in dists
    }

    # Run estimation
    for n_samples_per_est in n_samples_list:
        for name, dist in dists.items():
            # Mockup: replace this with your actual function
            true_pdf_value, mean_est, cof = test_estimate_pdf_value_from_sample(dist, n_samples_per_est)

            # Save results
            results[name]["n_samples"].append(n_samples_per_est)
            results[name]["true_pdf_value"].append(true_pdf_value)
            results[name]["mean_est"].append(mean_est)
            results[name]["cov"].append(cof)

    # Plotting results:
    _, axes = plt.subplots(3, 1, figsize=(10, 6 * 3))

    colour_list = ["r", "g", "b", "y", "m", "c", "k"]
    for (name, data), c in zip(results.items(), colour_list, strict=False):
        n_samples = data["n_samples"]
        true_pdf = np.array(data["true_pdf_value"])
        mean_est = np.array(data["mean_est"])
        cov = data["cov"]

        # Plot true_pdf_value and mean_est
        axes[0].plot(n_samples, true_pdf, label=f"{name} (True pdf values)", linestyle="--", c=c)
        axes[0].plot(n_samples, mean_est, label=f"{name} (estimator mean est)", c=c)
        axes[0].set_xlabel("Number of Samples")
        axes[0].set_ylabel("pdf")

        # proportion of estimate
        axes[1].plot(n_samples, mean_est / true_pdf, label=f"{name} mean_est/true", linestyle="--", c=c)
        axes[1].set_xlabel("Number of Samples")
        axes[1].set_ylabel("estimator_bias/ true value")

        # coefficient of variation
        axes[2].plot(n_samples, cov, label=f"{name} cov")
        axes[2].set_xlabel("Number of Samples")
        axes[2].set_ylabel("coeffecint of variation")


# ---- Tests for sample_quantile_estimate ----
class TestSampleQuantileEstimateUnit:
    """Unit tests checking the structure and basic properties of sample_quantile_estimate."""

    def test_median_peak_at_middle_sample(self):
        """For quantile=0.5, the peak probability should be near the median sample."""
        samples = torch.arange(1.0, 11.0)  # [1, 2, ..., 10]
        result = sample_quantile_estimate(samples, 0.5)
        peak_idx = result.pk.argmax()  # pyright: ignore[reportAttributeAccessIssue]
        # For n=10, q=0.5: mode of Binomial(10, 0.5) is k=5, so X_{(5)} = 5.0
        assert result.xk[peak_idx] == pytest.approx(5.0)  # pyright: ignore[reportAttributeAccessIssue]

    def test_high_quantile_peak_at_high_sample(self):
        """For quantile=0.9, the peak should be near the 90th percentile sample."""
        samples = torch.arange(1.0, 11.0)
        result = sample_quantile_estimate(samples, 0.9)
        peak_idx = result.pk.argmax()  # pyright: ignore[reportAttributeAccessIssue]
        # Mode of Binomial(10, 0.9) is k=9, so X_{(9)} = 9.0
        assert result.xk[peak_idx] == pytest.approx(9.0)  # pyright: ignore[reportAttributeAccessIssue]

    def test_low_quantile_peak_at_low_sample(self):
        """For quantile=0.1, the peak should be near the 10th percentile sample."""
        samples = torch.arange(1.0, 11.0)
        result = sample_quantile_estimate(samples, 0.1)
        peak_idx = result.pk.argmax()  # pyright: ignore[reportAttributeAccessIssue]
        # Mode of Binomial(10, 0.1) is k=1, so X_{(1)} = 1.0
        assert result.xk[peak_idx] == pytest.approx(1.0)  # pyright: ignore[reportAttributeAccessIssue]

    def test_cdf_and_ppf_are_consistent(self):
        """CDF and PPF should be inverses (at support points)."""
        samples = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        result = sample_quantile_estimate(samples, 0.5)
        # ppf(cdf(x)) should return x for support points
        for x in [1.0, 2.0, 3.0, 4.0, 5.0]:
            cdf_val = result.cdf(x)
            recovered = result.ppf(cdf_val)
            assert recovered == pytest.approx(x)

    def test_docstring_example(self):
        """Verify the worked example in the docstring.

        samples = [1, 2, 3, 4, 5], quantile = 0.2
        For s_i = 1 (k=1): Binomial(n=5, p=0.2).pmf(1)
        """
        samples = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        result = sample_quantile_estimate(samples, 0.2)

        # Check probability assigned to s_i=1 (k=1) matches Binomial(5, 0.2).pmf(1)
        expected = np.array([binom.pmf(k, 5, 0.2) for k in range(1, 6)])
        np.testing.assert_allclose(result.pk[1:], expected, atol=1e-10)  # pyright: ignore[reportAttributeAccessIssue]


@pytest.mark.non_deterministic
@pytest.mark.parametrize(
    "true_dist, quantile",
    [
        # Median (q=0.5) - well behaved, should be well calibrated
        (Gumbel(3, 2), 0.5),
        (Uniform(3, 5), 0.5),
        (Weibull(3, 2), 0.5),
        # Moderate quantiles
        (Gumbel(3, 2), 0.25),
        (Gumbel(3, 2), 0.75),
        # Higher quantiles (tail behaviour)
        (Gumbel(3, 2), 0.9),
        (Gumbel(3, 2), 0.95),
    ],
)
def test_sample_quantile_estimate_calibration(true_dist: Distribution, quantile: float):
    """Check that confidence bounds from sample_quantile_estimate are well-calibrated.

    Approach:
        1. Draw many sets of samples from a known distribution.
        2. For each sample set, build the quantile estimate distribution.
        3. Check if the true quantile value falls in the estimated confidence 95% interval.
        4. If calibrated, the true quantile should fall outside the 95% bounds ~5% of the time.

    Note:
        Tolerance is generous (abs=0.04) because the discrete nature of rv_discrete
        and the finite sample approximation introduce some miscalibration.
    """
    n_samples_per_est = 1000
    n_ests = 3000

    true_quantile_value = float(true_dist.icdf(tensor(quantile)))

    samples = true_dist.sample(torch.Size([n_ests, n_samples_per_est]))

    contains_count = 0
    for sample in samples:
        est_dist = sample_quantile_estimate(sample, quantile)
        lower = est_dist.ppf(0.025)
        upper = est_dist.ppf(0.975)
        if lower <= true_quantile_value <= upper:
            contains_count += 1

    coverage = contains_count / n_ests

    assert coverage == pytest.approx(0.95, abs=0.04), f"Expected ~95% coverage, got {coverage:.1%}"


@pytest.mark.non_deterministic
def test_sample_quantile_estimate_improves_with_more_samples():
    """Confidence interval width should decrease as sample size increases."""
    true_dist = Gumbel(3, 2)
    quantile = 0.5
    n_ests = 500

    widths_by_n = {}
    for n_samples in [20, 50, 200]:
        samples = true_dist.sample(torch.Size([n_ests, n_samples]))
        widths = []
        for sample in samples:
            est_dist = sample_quantile_estimate(sample, quantile)
            width = est_dist.ppf(0.975) - est_dist.ppf(0.025)
            widths.append(width)
        widths_by_n[n_samples] = np.mean(widths)

    # CI width should strictly decrease with more samples
    assert widths_by_n[20] > widths_by_n[50] > widths_by_n[200], f"CI widths should decrease: {widths_by_n}"
