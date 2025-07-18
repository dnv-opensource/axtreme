"""This module provides methods for importance sampling.

Importance sampling focuses computational effort on regions of interest. This is especially useful when only a small
part of the environment data contributes meaningfully to the quantity of interest (QoI).

In this file the following is included:
- Create importance sample and weights for a given importance distribution.
- Create importance sample and weights for a uniform region.

TODO(sw 25-05-26): This should be moved to src/axtreme/sampling once sufficiently tested.
"""

from collections.abc import Callable

import torch

torch.set_default_dtype(torch.float64)


# TODO(sw25-05-26): make he docstring here a bit clearer. Bit more explanation about the input functions.
# Maybe change the order
def importance_sampling_from_distribution(
    env_distribution_pdf: Callable[[torch.Tensor], torch.Tensor],
    importance_distribution_pdf: Callable[[torch.Tensor], torch.Tensor],
    importance_sampling_sampler: Callable[[torch.Size], torch.Tensor],
    num_samples: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Perform importance sampling for a given importance distribution.

    This function generates samples from a given importance sampling distribution and computes weights
    that accounts for the relation between the true environment distribution and the importance distribution, i.e.,

    - Suppose p(x) is the probability density function (PDF) of the true environment.
    - Suppose h(x) is the PDF of the importance sampling distribution.
    - Then, for each sample x, the importance sampling weight is: w(x) = p(x) / h(x).

    Args:
        env_distribution_pdf: The pdf function of the real environment distribution. This can either be a known function
            or for example be estimated based on the environment data using KDE (kernel density estimation).
        importance_distribution_pdf: The pdf function of the importance sampling distribution.
        importance_sampling_sampler: The sampler for producing samples from the importance sampling distribution.
        num_samples: The number of samples to generate.

    Returns:
        samples: The samples generated from the importance sampling distribution.
        weights: The importance sampling weights for the samples.

    Notes:
        A theoretical requirement for the importance sampling distribution is that it should be non-zero where the real
        environment distribution is non-zero. This means that: p(x) != 0 => h_x(x) != 0.
        However, in practice, this requirement is often relaxed.
    """
    samples = importance_sampling_sampler(torch.Size([num_samples]))
    pdf = env_distribution_pdf(samples)
    importance_pdf = importance_distribution_pdf(samples)

    weights = pdf / importance_pdf
    return samples, weights


# TODO(sw25-05-26): make the docstring here a bit clearer
# num_samples_total; this should probably define the number of samples to return, and then continue generating them
# we get enough. (This will also need to track some summary statistics)
def importance_sampling_distribution_uniform_region(
    env_distribution_pdf: Callable[[torch.Tensor], torch.Tensor],
    region: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    threshold: float,
    num_samples_total: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generates the importance sampling samples and weights by sampling uniformly from a defined region of interest.

    This method is useful when the importance distribution is unknown. The method constructs an implicit importance
    distribution by:
    1. Sampling uniformly in a broad region.
    2. Filtering samples based on whether their environment PDF exceeds a threshold.
    3. Estimating a uniform importance distribution over the filtered subset.
    4. Calculating weights as w(x) = p(x) / h(x), where h(x) is approximated analytically.

    Args:
        env_distribution_pdf: The pdf function of the real environment distribution.
            It should be callable with a tensor of shape (num_samples, d) and return a tensor of shape (num_samples,)
            Where d is the size of the input space.
        region: The bounds of the region to generate samples from. Can be a tuple of two tensors or a single tensor.

        if a single tensor:

          - shape: (2, d) where the first tensor is the lower bounds and the second tensor is the upper bounds.

        if a tuple of two tensors:

          - the first tensor is the lower bounds and the second tensor is the upper bounds.
          - both with shape: (d,)

        threshold: Environment regions with pdf values less than this threshold will not be explored by the importance
            samples. See `Details` for more information on how this threshold is used.
        # TODO(ak-06-10): see comment above by Sebastian: num_samples_total should equal number of returned samples
        num_samples_total: Total number of samples to draw uniformly before filtering. The actual number of
                           returned samples may be smaller depending on how many pass the threshold filter.

    Returns:
        A tuple (Tensor, Tensor) containing:
            The filtered samples drawn from the uniform distribution. Shape (n_samples,d)
            Importance sampling weights for each sample. Shape (n_samples,)

    Details:
        The algorithm works as follows:
        1. For a chosen `threshold` c for the real environment distribution.

           1.1 Define a sub-region F = {x: p(x) > c}, where p(x) is the probability density function
                of the real environment distribution. This should contain the significant regions of p(x).

        2. Generate `num_samples_total` uniform samples from the region. The region must cover all of F.

           2.1 Discard any points not in F. `num_samples` is the number of points that are left after discarding.

        3. The PDF of the sampled points `h_x(x)` is a uniform distribution over the region F.

           3.1 `h_x(x)` is estimated with `1/volume(region) * num_samples_total/num_samples`.

        4. The importance sampling weights are then calculated as w(x) = p(x)/h_x(x)

        This importance sampling approach is useful when:
            - We want to find a tighter/smaller region for our problem automatically. This approach can also
            automatically create non-linear boundaries.
            - You may have disjoint regions.
            - Need the impact of rare events to be explored.

        The approach is sensitive to the following parameters. Often trial and and error (and visualisation) is required
        to find an appropriate combination:
        - `threshold`:

          - Too low: Areas of the environment space that are practically impossible (and therefore irrelevant) will be
            included. This will create pointless additional computation.
          - Too high: Areas of the environment space that are important for down stream calculations may be excluded.
            (e.g rare but important event). Calculations done with this dataset could then give incorrect results.

        - `region`:

          - Too large: num_samples_total will need to be very large in order to get enough samples in F.
          - Too small: Areas of the environment space that are important for down stream calculations may be excluded.
            (e.g rare but important event). Calculations done with this dataset could then give incorrect results.

        The following check can help determine the suitability of you parameters:
        - Plot and check with a domain expert that the important regions of space are in F. Use a larger F if unsure.
        - Ensure that you are left with enough samples in F.

        Notes: This implementation breaks the importance sampling assumption that `p(x_i) != 0 => h_x(x_i) != 0`.
            - This still produces an exact result when responses/downstream-task [lets call it r(x_i)] is 0.
                - importance_result = sum[ r(x_i) * p(x_i)/h_x(x_i)] where x ~ h_x(x)
                - The point would add 0 to the non-importance weighted sum.
            - It will produce an approximate result if r(x_i) != 0.
                - This is a reasonable approximation if p(x_i) is considered to be close enough to 0.

    Todo: TODO
    - (sw 2024_09_17): Reference the paper here (or preprint) once its published Issue #177.

    """
    uniform_dist = torch.distributions.Uniform(region[0], region[1])

    # Generate samples from the uniform distribution over the region
    samples = uniform_dist.sample(torch.Size([num_samples_total]))

    # Calculate the probability density of the samples
    pdf = env_distribution_pdf(samples)

    # Find the samples that are above the threshold
    mask = pdf > threshold
    samples = samples[mask]

    # Calculate the volume of the hyper rectangle that contains the samples
    volume = torch.prod(region[1] - region[0])

    # The number of samples that are above the threshold
    num_samples = samples.shape[0]

    # Calculate the importance sampling distribution
    # The importance sampling distribution is estimated to be
    # h_x(x) = num_samples_total/(volume(region) * num_samples)
    h_x = num_samples_total / (volume * num_samples)

    # Calculate the importance sampling weights
    weights = pdf[mask] / h_x

    return samples, weights
