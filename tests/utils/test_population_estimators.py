# %%
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from scipy.stats import t as scipy_student_t
from torch.distributions import Distribution, Gumbel, StudentT, Uniform, Weibull

from axtreme.utils.population_estimators import estimate_pdf_value_from_sample, sample_mean_se, sample_median_se


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
def test_sample_mean_se(true_dist: Distribution, n_samples_per_est: int = 20, n_ests: int = 5000):
    """Checks if the distribution returned by sample_mean_se is reasonable.

    The consifidence bounds returned by this method are frequently used. We want to check that they are well calibrated.
    If we obtain at 95% confidence from a sample, 95% of the time the true mean should fall within these bounds. Higher
    or lower percentage represent miscallibrate.

    NOTE: sample_mean_se is primarily used as a diagnoistic, so we are willing to accept some error in the results.

    Args:
        true_dist: The true ditribution from which to generate samples from. The method has no restriction of the type
          of sample distribution.
    """
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
def test_sample_median_se(true_dist: Distribution, n_samples_per_est: int = 50, n_ests: int = 5000):
    """Checks if the distribution returned by sample_median_se is reasonable.

    The consifidence bounds returned by this method are frequently used. We want to check that they are well calibrated.
    If we obtain at 95% confidence from a sample, 95% of the time the true mean should fall within these bounds. Higher
    or lower percentage represent miscallibrate.

    NOTE: sample_median_se is primarily used as a diagnoistic, so we are willing to accept some error in the results.

    Args:
        true_dist: The true ditribution from which to generate samples from. The method has no restriction of the type
          of sample distribution.
    """
    true_median = true_dist.icdf(0.5)

    samples = true_dist.sample(torch.Size([n_ests, n_samples_per_est]))

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
def test_estimate_pdf_value_from_sample_at_median(true_dist: Distribution, n_ests: int = 5000):
    """This tests for degredation in the performance on estimate_pdf_value_from_sample_at_median.

    The bounds required depened on `n_samples_per_est`. `visualise_performance_of_estimate_pdf_value_from_sample` can be
    used to determine them.
    """
    n_samples_per_est = 50

    q = 0.5
    x = true_dist.icdf(q)
    true_pdf_value = true_dist.log_prob(x).exp()

    samples = true_dist.sample(torch.Size([n_ests, n_samples_per_est]))

    pdf_ests = [estimate_pdf_value_from_sample(sample, x) for sample in samples]

    results = torch.tensor(pdf_ests)

    assert float(results.mean()) == pytest.approx(true_pdf_value, rel=0.2)
    coeffeceint_of_variation = results.std() / results.mean()
    assert coeffeceint_of_variation < 0.2


def visualise_performance_of_estimate_pdf_value_from_sample():
    """Show the bias and variance of `estimate_pdf_value_from_sample` an n_samples increases"""

    def test_estimate_pdf_value_from_sample(true_dist: Distribution, n_samples_per_est: int = 50, n_ests: int = 5000):
        q = 0.5
        x = true_dist.icdf(q)
        true_pdf_value = true_dist.log_prob(x).exp()

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
    results = {name: {"n_samples": [], "true_pdf_value": [], "mean_est": [], "cov": []} for name in dists}

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
