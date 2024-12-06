"""Helpers for understanding the population values expected by estimators.

NOTE: These tool provide indicative/approximate result.
"""

from typing import Any

import matplotlib.pyplot as plt
import torch
from matplotlib.axes import Axes
from scipy.stats import gaussian_kde
from torch.distributions import Distribution, Normal, StudentT


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
