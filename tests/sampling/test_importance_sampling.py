"""
Test the creation of importance sampling.

The theory is derived and explained in:

Winter, Sebastian, Christian Agrell, Juan Camilo Guevara Gómez, and Erik Vanem. “Efficient Long-Term Structural
Reliability Estimation with Non-Gaussian Stochastic Models: A Design of Experiments Approach.” arXiv, March 3, 2025.
https://doi.org/10.48550/arXiv.2503.01566.

and is therefore not part of the tests. The focus here is on unit tests of the two functions used to create the
importance samples and weights for a given environment distribution.
"""

from unittest.mock import patch

import torch

from axtreme.sampling.importance_sampling import (
    importance_sampling_distribution_uniform_region,
    importance_sampling_from_distribution,
)


def test_importance_sampling_from_distribution():
    """Basic test to see if the function runs and returns the expected output with simple Callables."""

    samples = torch.tensor([1.0, 2.0, 3.0, 4.0])

    def _mock_sampler(size: torch.Size) -> torch.Tensor:
        return samples[: size[0]]

    def _env_distribution_pdf(x: torch.Tensor) -> torch.Tensor:
        return x + 1

    def _importance_distribution_pdf(x: torch.Tensor) -> torch.Tensor:
        return x + 2

    samples, weights = importance_sampling_from_distribution(
        env_distribution_pdf=_env_distribution_pdf,
        importance_distribution_pdf=_importance_distribution_pdf,
        importance_sampling_sampler=_mock_sampler,
        num_samples=3,
    )

    assert torch.equal(samples, torch.tensor([1.0, 2.0, 3.0]))
    assert torch.equal(weights, torch.tensor([2.0 / 3.0, 3.0 / 4.0, 4.0 / 5.0]))


def test_importance_sampling_distribution_uniform_region():
    """Basic test to see if the function runs and returns the expected output with simple Callables."""

    def _env_distribution_pdf(x: torch.Tensor) -> torch.Tensor:
        return x

    region = (torch.tensor([0.0]), torch.tensor([4.0]))

    threshold = 2.0
    num_samples_total = 6
    fixed_samples = torch.tensor([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])

    # Temporarily replace the .sample() method of torch.distributions.Uniform with deterministic version that always
    # returns the predefined fixed_samples during the with block. This is necessary as no seeding is implemented for
    # the function importance_sampling_distribution_uniform_region.
    with patch.object(torch.distributions.Uniform, "sample", return_value=fixed_samples):
        samples, weights = importance_sampling_distribution_uniform_region(
            env_distribution_pdf=_env_distribution_pdf,
            region=region,
            threshold=threshold,
            num_samples_total=num_samples_total,
        )

    assert torch.equal(samples, torch.Tensor([2.5, 3.0]))
    assert torch.equal(weights, torch.Tensor([2.5 * 4 / 3, 3.0 * 4 / 3]))
