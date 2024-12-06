import pytest
import torch
from botorch.posteriors.gpytorch import GPyTorchPosterior
from tests.sampling.helpers import (
    posterior_b2_n3_t2,
    posterior_n1_t1,
    posterior_n1_t2,
    posterior_n2_t1,
    posterior_n3_t2,
)

from axtreme.sampling import UTSampler


@pytest.mark.parametrize(
    "posterior, expected_base_sample_shape",
    [
        (posterior_n1_t1(), torch.Size([3, 1])),  # MultivariateNormal
        (posterior_n1_t2(), torch.Size([5, 2])),  # MultitaskMultivariateNormal
    ],
)
def test_construct_base_samples_created_with_right_shape(
    posterior: GPyTorchPosterior, expected_base_sample_shape: torch.Size
):
    """NOTE: the sample_shape for UT is determined by the size of the input.

    sample_shape = 2 * t + 1, where t comes from the posterior shape (*b,n,t)
    """
    sampler = UTSampler()
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
    sampler = UTSampler()
    sampler._construct_base_samples(posterior1)
    original_base_samples = id(sampler.base_samples)

    sampler._construct_base_samples(posterior2)
    base_samples_after_repeat = id(sampler.base_samples)

    # The same obect should be used
    assert original_base_samples == base_samples_after_repeat


def test_construct_base_samples_for_posterior_with_new_target_size():
    sampler = UTSampler()
    sampler._construct_base_samples(posterior_n1_t1())
    base_samples_1 = sampler.base_samples
    sampler._construct_base_samples(posterior_n1_t2())
    base_samples_2 = sampler.base_samples

    assert base_samples_1.shape == torch.Size([3, 1])
    assert base_samples_2.shape == torch.Size([5, 2])


@pytest.mark.parametrize(
    "posterior, expected_sample_shape",
    [
        (posterior_n1_t1(), torch.Size([3, 1, 1])),  # MultivariateNormal
        (posterior_n1_t2(), torch.Size([5, 1, 2])),  # MultitaskMultivariateNormal
        (posterior_n3_t2(), torch.Size([5, 3, 2])),  # MultitaskMultivariateNormal
        (posterior_b2_n3_t2(), torch.Size([5, 2, 3, 2])),  # MultitaskMultivariateNormal batched
    ],
)
def test_forward(posterior: GPyTorchPosterior, expected_sample_shape: torch.Size):
    sampler = UTSampler()
    samples = sampler(posterior)

    assert samples.shape == expected_sample_shape


@pytest.mark.parametrize(
    "posterior, sample_shape, expected_mean_shape",
    [
        (posterior_n1_t1(), (3, 1), (1,)),  # MultivariateNormal
        (posterior_n1_t1(), (3, 2, 2), (2, 2)),  # MultivariateNormal
        (posterior_n1_t2(), (5, 3), (3,)),  # MultitaskMultivariateNormal
        (posterior_b2_n3_t2(), (5, 4), (4,)),  # MultitaskMultivariateNormal batched
    ],
)
def test_mean_shape(posterior: GPyTorchPosterior, sample_shape: torch.Size, expected_mean_shape: torch.Size):
    sampler = UTSampler()
    sampler._construct_base_samples(posterior)
    samples = torch.randn(sample_shape)
    mean = sampler.mean(samples, dim=0)

    assert mean.shape == expected_mean_shape


@pytest.mark.parametrize(
    "posterior, dim, sample_shape, expected_mean_shape",
    [
        (posterior_n1_t1(), 0, (3, 1), (1,)),  # MultivariateNormal
        (posterior_n1_t1(), 1, (2, 3, 2), (2, 2)),  # MultivariateNormal
        (posterior_n1_t2(), 1, (3, 5), (3,)),  # MultitaskMultivariateNormal
        (posterior_b2_n3_t2(), 2, (1, 1, 5), (1, 1)),  # MultitaskMultivariateNormal
    ],
)
def test_mean_different_dim(
    posterior: GPyTorchPosterior,
    dim: int,
    sample_shape: torch.Size,
    expected_mean_shape: torch.Size,
):
    sampler = UTSampler()
    sampler._construct_base_samples(posterior)
    samples = torch.randn(sample_shape)
    mean = sampler.mean(samples, dim=dim)

    assert mean.shape == expected_mean_shape


@pytest.mark.parametrize(
    "posterior, dim, sample_shape",
    [
        (posterior_n1_t1(), 0, (1, 1)),  # MultivariateNormal
        (posterior_n1_t1(), 1, (3, 1)),  # MultivariateNormal
        (posterior_n1_t2(), 0, (3, 5)),  # MultitaskMultivariateNormal
        (posterior_b2_n3_t2(), 1, (5, 4)),  # MultitaskMultivariateNormal batched
    ],
)
def test_mean_fail_when_wrong_shape(
    posterior: GPyTorchPosterior,
    dim: int,
    sample_shape: torch.Size,
):
    sampler = UTSampler()
    sampler._construct_base_samples(posterior)
    samples = torch.randn(sample_shape)
    with pytest.raises(ValueError):
        _ = sampler.mean(samples, dim=dim)


@pytest.mark.parametrize(
    "posterior",
    [
        (posterior_n1_t1()),  # MultivariateNormal
        (posterior_n1_t2()),  # MultitaskMultivariateNormal
        (posterior_b2_n3_t2()),  # MultitaskMultivariateNormal batched
    ],
)
def test_mean_value(posterior: GPyTorchPosterior):
    sampler = UTSampler()
    samples = sampler(posterior)
    mean = sampler.mean(samples)
    posterior_mean = posterior.mean

    assert torch.allclose(mean, posterior_mean, atol=1e-10)


@pytest.mark.parametrize(
    "posterior, sample_shape, expected_var_shape",
    [
        (posterior_n1_t1(), (3, 1), (1,)),  # MultivariateNormal
        (posterior_n1_t1(), (3, 2, 2), (2, 2)),  # MultivariateNormal
        (posterior_n1_t2(), (5, 3), (3,)),  # MultitaskMultivariateNormal
        (posterior_b2_n3_t2(), (5, 4), (4,)),  # MultitaskMultivariateNormal batched
    ],
)
def test_var_shape(
    posterior: GPyTorchPosterior,
    sample_shape: torch.Size,
    expected_var_shape: torch.Size,
):
    sampler = UTSampler()
    sampler._construct_base_samples(posterior)
    samples = torch.randn(sample_shape)
    var = sampler.var(samples, dim=0)

    assert var.shape == expected_var_shape


@pytest.mark.parametrize(
    "posterior, dim, sample_shape, expected_var_shape",
    [
        (posterior_n1_t1(), 0, (3, 1), (1,)),  # MultivariateNormal
        (posterior_n1_t1(), 1, (2, 3, 2), (2, 2)),  # MultivariateNormal
        (posterior_n1_t2(), 1, (3, 5), (3,)),  # MultitaskMultivariateNormal
        (posterior_b2_n3_t2(), 2, (1, 1, 5), (1, 1)),  # MultitaskMultivariateNormal batched
    ],
)
def test_var_different_dim(
    posterior: GPyTorchPosterior,
    dim: int,
    sample_shape: torch.Size,
    expected_var_shape: torch.Size,
):
    sampler = UTSampler()
    sampler._construct_base_samples(posterior)
    samples = torch.randn(sample_shape)
    var = sampler.var(samples, dim=dim)

    assert var.shape == expected_var_shape


@pytest.mark.parametrize(
    "posterior, dim, sample_shape",
    [
        (posterior_n1_t1(), 0, (1, 1)),  # MultivariateNormal
        (posterior_n1_t1(), 1, (3, 1)),  # MultivariateNormal
        (posterior_n1_t2(), 0, (3, 5)),  # MultitaskMultivariateNormal
        (posterior_b2_n3_t2(), 1, (5, 4)),  # MultitaskMultivariateNormal batched
    ],
)
def test_var_fail_when_wrong_shape(
    posterior: GPyTorchPosterior,
    dim: int,
    sample_shape: torch.Size,
):
    sampler = UTSampler()
    sampler._construct_base_samples(posterior)
    samples = torch.randn(sample_shape)
    with pytest.raises(ValueError):
        _ = sampler.var(samples, dim=dim)


@pytest.mark.parametrize(
    "posterior",
    [
        (posterior_n1_t1()),  # MultivariateNormal
        (posterior_n1_t2()),  # MultitaskMultivariateNormal
        (posterior_b2_n3_t2()),  # MultitaskMultivariateNormal batched
    ],
)
def test_var_value(posterior: GPyTorchPosterior):
    sampler = UTSampler()
    samples = sampler(posterior)
    var = sampler.var(samples)
    posterior_var = posterior.variance

    assert torch.allclose(var, posterior_var, atol=1e-10)
