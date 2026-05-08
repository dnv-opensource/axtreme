# %%
"""Helpers for understanding the population values expected by estimators.

NOTE: These tool provide indicative/approximate result. These are currently considered experimental.
"""

from typing import Any

import matplotlib.pyplot as plt
import torch
from matplotlib.axes import Axes
from scipy.stats import gaussian_kde, rv_discrete
from torch.distributions import Binomial, Distribution, Normal, StudentT


def sample_mean_se(samples: torch.Tensor) -> StudentT:
    """Distibution of the population mean as estimated by this sample.

    Note:
        The distribution of the sample itself doesn't matter. The output distibution is not effected by this.

    Use the following `link <https://stats.libretexts.org/Courses/Lake_Tahoe_Community_College/Book%3A_Introductory_Statistics_(OpenStax)_With_Multimedia_and_Interactivity_LibreTexts_Calculator/08%3A_Confidence_Intervals/8.03%3A_A_Single_Population_Mean_using_the_Student_t-Distribution>`_

    se = sigma / n**.5

        - Sigma: should be the population standard deviation, but we approximate this with the sample standard deviation
        - Because of this approximation we use the Student-t distribution

    Args:
        samples: 1d tensor of sample to estimate the population mean from

    Return:
        Distribution of the population mean based on the provided sample. Additionally, it provides the 95% confidence
        bounds for the estimate, which are typically calculated to fall within a range of 93% to 97% coverage, depending
        on sample variability and the assumptions of the calculation method.

    Todo:
        - .cdf() raises NotImplementedError for torch implemenation of StudentT. This is annoying
            because this is the best way to check the confidence bounds (using z = (y - mean)/stddev) assumes you
            are using a normal distibution rather than a student t distibution. This approximation is considered okay
            for n>30)
    """
    sample_mean = samples.mean()
    # Note by default torch.std() give the sample std: 1/(n-1) * sum[(x_i - x_bar)**2]
    sample_mean_se = samples.std() / len(samples) ** 0.5
    return StudentT(df=len(samples) - 1, loc=sample_mean, scale=sample_mean_se)


def sample_median_se(samples: torch.Tensor) -> Normal:
    """Distibution of the population median as estimated by this sample.

    Details of this method can be found `here. <https://en.wikipedia.org/wiki/Median#Sampling_distribution.>`_

    Note:
        This function relies on the approximation `estimate_pdf_value_from_sample`. The approximation can be quite
        inaccurate (see the function for details), and as a result this function should be treated as an estimate.

    Note:
        The result returned are much more noisey than sample_mean_se.

    Args:
        samples: 1d tensor of sample to estimate the population median from

    Return:
        Distibution of the population median as estimated by this sample. The 95% bounds (with 50 samples) estimated by
        this function typically produce bounds actually between 90%-97%.
    """
    sample_median = samples.median()
    # pdf at median point
    f_m = estimate_pdf_value_from_sample(samples, float(sample_median))

    sample_median_se = (1 / (4 * len(samples) * f_m**2)) ** 0.5
    return Normal(loc=sample_median, scale=sample_median_se)


def estimate_pdf_value_from_sample(sample: torch.Tensor, x: float) -> float:
    """Construct a distibution from a sample, and get the pdf value at point x.

    WARNING: This is an approximate method, and results impove with more samples. See testing results below.

    Args:
        sample: 1d Samples to construct a pdf from.
        x: the point at which to evaluate the pdf.

    Return:
        Estimated pdf value.

    Testing results:
    The mean returned and the cof have the following behaviour. Full test detail can be run at
    `tests/utils/test_population_estimators.py : visualise_performance_of_estimate_pdf_value_from_sample`

    Number of samples | mean_est/true |   Coef   |
    ==============================================
    11                |  .86  - 1.02  | .25-.30  |
    22                |  .88  - 1.02  | .18-.20  |
    44                |  .90  - 1.01  | .14-.16  |
    88                |  .92  - 1.00  | .11-.13  |
    176               |  .95  - 1.00  | .05-.10  |
    """
    kde = gaussian_kde(sample.numpy())
    return float(kde(x)[0])


def plot_dist(
    dist: Distribution,
    confidence_level: float = 3.0,
    ax: Axes | None = None,
    **kwargs: Any,  # noqa: ANN401
) -> Axes:
    """Plot the distribution PDF over domain `mean +- confidence_level * std`.

    Args:
        dist: the distribution to plot the pdf of.
        confidence_level: controls the width of the plot
        ax: the axes to plot on. If None, will create an x
        **kwargs: passed to the plotting method.

    Return:
        Axes with the plot.
    """
    if ax is None:
        ax = plt.subplot()

    x = torch.linspace(dist.mean - confidence_level * dist.stddev, dist.mean + confidence_level * dist.stddev, 100)

    _ = ax.plot(x, dist.log_prob(x).exp(), **kwargs)

    return ax


def sample_quantile_estimate(samples: torch.Tensor, quantile: float) -> rv_discrete:
    r"""Returns an analytical distribution over which sample value is the true population quantile.

    For each sorted sample value $s_i$, computes the likelihood that $s_i$ is the true population
    quantile $q$ using a Binomial model.

    Details
    - For each sample $s_i$ provided, estimate the p(data | cdf_true(s_i) = q), where:
        - cdf_true(s_i) = Prb(X <= s_i) = q
        - (number_of_samples <= s_i) = k
        - Can be framed as a binomial distribution: Given drew $n$ samples, and the prb of being <= s_i is $q$, what is
           the prb of seeing $k$ samples <= s_i?
    - p(data | cdf_true(s_i) = q) = Binomial(n = len(samples), k = number of samples <= s_i, p = quantile)

    - Example:
        For ``samples = [1, 2, 3, 4, 5]`` and ``quantile = 0.2``:

        - For $s_i = 1$: $k = 1$, so ``Binomial(n=5, p=0.2).pmf(1)``
        - For $s_i = 2$: $k = 2$, so ``Binomial(n=5, p=0.2).pmf(2)``
        - For $s_i = 3$: $k = 3$, so ``Binomial(n=5, p=0.2).pmf(3)``
        - etc.

    - NOTE: $k$ counts samples $\leq s_i$ (not $< s_i$) because the CDF is defined as $P(X \leq x)$.

    - NOTE: The case $k=0$ (true quantile is below all samples) can't be represented with the true samples.
        We artificially add a point to associate this mass with.
        - `samples = [.98,1, 2, 3, 4, 5]`` and ``quantile = 0.2``:
        - For $s_i = .98$: $k = 0$, so ``Binomial(n=5, p=0.2).pmf(0)``
        Benefits:
            - discrete distribution need prb mass to sum to 1. Alternative is to renormalise, but cdf calculation would
              then underestimate q by up to ``Binomial(n, p).pmf(0)``.
        Cons:
            - Artificial datapoint has now been added. This will affect mean/var etc calculation if performed.

    Example usage:
        >>> samples = torch.tensor([1, 2, 3, 4, 5])
        >>> quantile = 0.2
        >>> dist = sample_quantile_estimate(samples, quantile)
        >>> est_95_interval = (dist.ppf(0.025), dist.ppf(0.975))
        >>> plt.plot(dist.xk, dist.pk, marker="o")  # plot the locations and probabilities.

    Args:
        samples: 1D tensor of samples drawn from the population.
        quantile: The quantile of interest, in the range ``(0, 1)``.

    Returns:
        A discrete distribution whose support is the sorted sample values, weighted by the
        likelihood that each value equals the true population quantile. Can be used to obtain
        confidence intervals or the most likely quantile estimate.

    References:
        - https://en.wikipedia.org/wiki/Order_statistic#Application:_confidence_intervals_for_quantiles

    Todo:
        - TODO(sw 2026-4-25): Basic mathematics, and empirical validation (in unit tests) are done, but should find
            more rigorous references detailing the method.
        - TODO(sw 2026-4-25): Need to consider what happens at k = 0, and k=n.
            - k=0: We don't care about evaluating this point, but it is required to make a prb distribution
            - k=n: Distribution can't capture the possibility the could get samples.realisation larger than this
            - Would it be good to give warning when going out of bounds.
    """
    n = len(samples)
    samples = samples.clone().sort().values
    # Artificially add a point to capture prb mass at k=0. See Details for motivation.
    samples = torch.cat((samples[:1] * 0.98, samples))
    # need shape n+1 to capture k=0,...,n. Each of these will store the value n
    total_count = torch.full((n + 1,), n, dtype=torch.int64)
    dist = Binomial(total_count=total_count, probs=torch.tensor(quantile, dtype=torch.float64))
    # to capture k=0...n, need to pass n+1 items in
    log_prb = dist.log_prob(torch.arange(0, n + 1))
    prbs = log_prb.exp().numpy()

    rv_dist = rv_discrete(values=(samples.numpy(), prbs))
    return rv_dist
