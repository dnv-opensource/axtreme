import pytest
import torch
from botorch.posteriors.gpytorch import GPyTorchPosterior

from axtreme.sampling import NormalIndependentSampler
from tests.sampling.helpers import (
    posterior_b2_n3_t2,
    posterior_n1_t1,
    posterior_n1_t2,
    posterior_n2_t1,
    posterior_n3_t2,
)


### unit test
@pytest.mark.parametrize(
    "posterior, sample_shape ,expected_base_sample_shape",
    [
        (posterior_n1_t1(), torch.Size([7, 7]), torch.Size([7, 7, 1])),  # MultivariateNormal
        (posterior_n2_t1(), torch.Size([7, 7]), torch.Size([7, 7, 1])),  # MultivariateNormal multiple points
        (posterior_n3_t2(), torch.Size([7, 7]), torch.Size([7, 7, 2])),  # MultitaskMultivariateNormal
        # MultitaskMultivariateNormal with additional batching
        (posterior_b2_n3_t2(), torch.Size([7, 7]), torch.Size([7, 7, 2])),
    ],
)
def test_required_base_sample_shape(
    posterior: GPyTorchPosterior, sample_shape: torch.Size, expected_base_sample_shape: torch.Size
):
    """Check this works for different types of posteriors."""
    actual_shape = NormalIndependentSampler._required_base_sample_shape(sample_shape, posterior)
    assert expected_base_sample_shape == actual_shape


@pytest.mark.parametrize(
    "posterior, expected_base_sample_shape",
    [
        (posterior_n1_t1(), torch.Size([3, 1])),  # MultivariateNormal
        (posterior_n1_t2(), torch.Size([3, 2])),  # MultitaskMultivariateNormal
    ],
)
def test_construct_base_samples_created_with_right_shape(
    posterior: GPyTorchPosterior, expected_base_sample_shape: torch.Size
):
    sampler = NormalIndependentSampler(sample_shape=torch.Size([3]), seed=7)
    sampler._construct_base_samples(posterior)

    assert sampler.base_samples.shape == expected_base_sample_shape


@pytest.mark.parametrize(
    "posterior1, posterior2",
    [
        (posterior_n1_t1(), posterior_n2_t1()),  # MultivariateNormal
        (posterior_n1_t2(), posterior_n3_t2()),  # MultitaskMultivariateNormal
        (posterior_n1_t2(), posterior_b2_n3_t2()),  # MultitaskMultivariateNormal batched
    ],
)
def test_construct_base_samples_reuse_base_samples_when_possible(
    posterior1: GPyTorchPosterior, posterior2: GPyTorchPosterior
):
    """Reuse base samples if the samples shape and output dim t don't change."""
    sampler = NormalIndependentSampler(sample_shape=torch.Size([3]), seed=7)
    sampler._construct_base_samples(posterior1)
    original_base_samples = id(sampler.base_samples)

    sampler._construct_base_samples(posterior2)
    base_samples_after_repeat = id(sampler.base_samples)

    # The same obect should be used
    assert original_base_samples == base_samples_after_repeat


def test_construct_base_samples_for_posterior_with_new_target_size():
    sampler = NormalIndependentSampler(sample_shape=torch.Size([3]), seed=7)
    sampler._construct_base_samples(posterior_n1_t1())
    base_samples_1 = sampler.base_samples
    sampler._construct_base_samples(posterior_n1_t2())
    base_samples_2 = sampler.base_samples

    assert base_samples_1.shape == torch.Size([3, 1])
    assert base_samples_2.shape == torch.Size([3, 2])


def test_construct_base_samples_for_new_sample_shape():
    """NOTE: don't really expect this usecase, but it is supported."""
    sampler = NormalIndependentSampler(sample_shape=torch.Size([3]), seed=7)
    sampler._construct_base_samples(posterior_n1_t1())
    base_samples_1 = sampler.base_samples
    sampler.sample_shape = torch.Size([4, 4])
    sampler._construct_base_samples(posterior_n1_t1())
    base_samples_2 = sampler.base_samples

    assert base_samples_1.shape == torch.Size([3, 1])
    assert base_samples_2.shape == torch.Size([4, 4, 1])


@pytest.mark.parametrize(
    "posterior, sample_shape, expected_sample_shape",
    [
        (posterior_n1_t1(), torch.Size([1]), torch.Size([1, 1, 1])),  # MultivariateNormal
        (posterior_n1_t1(), torch.Size([3, 2]), torch.Size([3, 2, 1, 1])),  # MultivariateNormal
        (posterior_n1_t2(), torch.Size([3]), torch.Size([3, 1, 2])),  # MultitaskMultivariateNormal
        (posterior_n1_t2(), torch.Size([3, 2]), torch.Size([3, 2, 1, 2])),  # MultitaskMultivariateNormal
        (posterior_n3_t2(), torch.Size([5]), torch.Size([5, 3, 2])),  # MultitaskMultivariateNormal
        (posterior_n3_t2(), torch.Size([5, 2]), torch.Size([5, 2, 3, 2])),  # MultitaskMultivariateNormal
        (posterior_b2_n3_t2(), torch.Size([1]), torch.Size([1, 2, 3, 2])),  # MultitaskMultivariateNormal batched
        (posterior_b2_n3_t2(), torch.Size([5, 4]), torch.Size([5, 4, 2, 3, 2])),  # MultitaskMultivariateNormal batched
    ],
)
def test_forward(posterior: GPyTorchPosterior, sample_shape: torch.Size, expected_sample_shape: torch.Size):
    sampler = NormalIndependentSampler(sample_shape=sample_shape)
    samples = sampler(posterior)

    assert samples.shape == expected_sample_shape
