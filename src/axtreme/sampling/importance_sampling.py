# %%
"""This module provides methods to create importance samples and weights.

Importance sampling focuses computational effort on regions of interest. This is especially useful when only a small
part of the environment data contributes meaningfully to the quantity of interest (QoI).

In this file the following is included:
- Create importance sample and weights for a given importance distribution.
- Create importance sample and weights for a uniform region.

See docs/source/technical_details/importance_sampling.md and tutorials/importance_sampling.py for examples and guidance
on when and how to use importance sampling.
"""

from collections.abc import Callable

import torch
from torch.distributions.distribution import Distribution

torch.set_default_dtype(torch.float64)


def importance_sampling_from_distribution(
    env_distribution_pdf: Callable[[torch.Tensor], torch.Tensor],
    importance_distribution_pdf: Callable[[torch.Tensor], torch.Tensor],
    importance_sampling_sampler: Callable[[torch.Size], torch.Tensor],
    num_samples: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Perform importance sampling for a given importance distribution.

    Args:
        env_distribution_pdf: The pdf function of the real environment distribution.
        importance_distribution_pdf: The pdf function of the importance sampling distribution.
        importance_sampling_sampler: The sampler for producing samples from the importance sampling distribution.
        num_samples: The number of samples to generate.

    Returns:
        samples: The samples generated from the importance sampling distribution.
        weights: The importance sampling weights for the samples.

    Notes:
        A theoretical requirement for the importance sampling distribution is that it should be non-zero where the real
        environment distribution is non-zero. However, in practice, this requirement is often relaxed.
    """
    samples = importance_sampling_sampler(torch.Size([num_samples]))
    pdf = env_distribution_pdf(samples)
    importance_pdf = importance_distribution_pdf(samples)

    weights = pdf / importance_pdf
    return samples, weights


def importance_sampling_distribution_uniform_region(
    env_distribution_pdf: Callable[[torch.Tensor], torch.Tensor],
    region: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    threshold: float,
    num_samples_total: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generates importance samples and weights by uniformly sampling a region and filtering by env-pdf > threshold.

    The method constructs an implicit importance distribution by:
    1. Sampling uniformly in a broad region.
    2. Filtering samples based on whether their environment PDF (p(x)) exceeds a threshold.
    3. Estimating a uniform importance distribution, h(x), over the filtered subset.
    4. Calculating weights as w(x) = p(x) / h(x).

    The mathematical justification for this algorithm is given in:
    "Efficient Long-Term Structural Reliability Estimation with Non-Gaussian Stochastic Models: A Design of Experiments
    Approach.” arXiv, March 3, 2025. https://doi.org/10.48550/arXiv.2503.01566."

    Args:
        env_distribution_pdf: The pdf function of the real environment distribution.
            It should be callable with a tensor of shape (num_samples_total, d) and return a tensor of shape
            (num_samples_total,). Where d is the size of the input space.
        region: The bounds of the region to generate samples from. Can be a tuple of two tensors or a single tensor.

        if a single tensor:

          - shape: (2, d) where the first tensor is the lower bounds and the second tensor is the upper bounds.

        if a tuple of two tensors:

          - the first tensor is the lower bounds and the second tensor is the upper bounds.
          - both with shape: (d,)

        threshold: Environment regions with pdf values less than this threshold will not be explored by the importance
            samples.

        num_samples_total: Total number of samples to return.

    Returns:
        A tuple (Tensor, Tensor) containing:
            The filtered samples drawn from the uniform distribution. Shape (num_samples_total,d)
            Importance sampling weights for each sample. Shape (num_samples_total,)

    Notes:
        This implementation breaks the importance sampling assumption that `p(x_i) != 0 => h_x(x_i) != 0`.
        While h_x is non-zero everywhere in the defined region, it is zero outside of it.
        - This still produces an exact result when responses/downstream-task [lets call it r(x_i)] is 0.
            - importance_result = sum[ r(x_i) * p(x_i)/h_x(x_i)] where x ~ h_x(x)
            - The point would add 0 to the non-importance weighted sum.
            - It will produce an approximate result if r(x_i) != 0.
                - This is a reasonable approximation if p(x_i) is considered to be close enough to 0.
    """
    uniform_dist = torch.distributions.Uniform(region[0], region[1])

    def _create_samples_and_weights(
        dist: Distribution, num_samples_to_create: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Create samples and weights from a uniform distribution over a defined region."""
        # Generate samples from the uniform distribution over the region
        samples = dist.sample(torch.Size([num_samples_to_create]))

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
        h_x = num_samples_to_create / (volume * num_samples)

        # Calculate the importance sampling weights
        weights = pdf[mask] / h_x

        return samples, weights

    # Keep creating importance samples and weights until there are num_samples_total of them
    samples = torch.empty((0,))
    weights = torch.empty((0,))
    while len(samples) < num_samples_total:
        # If we were to create num_samples_total-len(samples) samples starting in the second iteration the weights would
        # be inconsistent due to the definition of h_x. Hence, we need to create too many samples and then take only
        # the needed amount of samples. This is computationally inefficient but as _create_samples_and_weights runs fast
        # this is acceptable.
        s, w = _create_samples_and_weights(uniform_dist, num_samples_total)
        num_missing_samples = min(num_samples_total - len(samples), len(s))
        samples = torch.cat((samples, s[:num_missing_samples]))
        weights = torch.cat((weights, w[:num_missing_samples]))

    return samples, weights
