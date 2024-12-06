# %%
import copy

import pytest
import torch
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.distributions.multivariate_normal import MultivariateNormal
from tests.sampling.helpers import (
    posterior_b2_n3_t2,
    posterior_n1_t1,
    posterior_n2_t1,
    posterior_n3_t2,
    posterior_n3_t2_interleaved,
)

from axtreme.sampling.independent_sampler import IndependentMCSampler, diagonalize_distribution


@pytest.mark.parametrize(
    "posterior",
    [
        (posterior_n1_t1()),  # test single input
        (posterior_n2_t1()),  # MultivariateNormal
        (posterior_n3_t2()),  # MultitaskMultivariateNormal
        (posterior_b2_n3_t2()),  # MultitaskMultivariateNormal batched
        (posterior_n3_t2_interleaved()),  # MultitaskMultivariateNormal interleaved
    ],
)
def test_diagonalize_distribution_remove_covariance(posterior: GPyTorchPosterior):
    ### set up the test
    # make a new posterior with covariance, by adding something to the off diagonal.
    # This is done by adding to the whole covariance matrix, and then setting the diagonal back to their original values
    posterior_with_covar = copy.deepcopy(posterior)
    _ = torch.add(posterior_with_covar.covariance_matrix, 0.01, out=posterior_with_covar.covariance_matrix)
    # reset the diagonals to their original
    n_points = posterior_with_covar.covariance_matrix.shape[-1]
    posterior_with_covar.covariance_matrix[..., range(n_points), range(n_points)] = (
        posterior.covariance_matrix.diagonal(dim1=-2, dim2=-1)
    )

    ### Run the code
    diag_posterior = diagonalize_distribution(posterior_with_covar)

    ### Check the output
    # There is not inbuilt equality for distibutions, so check the key components.
    torch.testing.assert_close(posterior.mean, diag_posterior.mean)
    torch.testing.assert_close(posterior.covariance_matrix, diag_posterior.covariance_matrix)
    assert posterior.base_sample_shape == diag_posterior.base_sample_shape
    # NOTE: we accept .lazy_covariance_matrix being different type (DiagLinearOperator, rather than DenseLinearOperator)

    # Check the attributes from the parent class pytorch.Distibutions.
    # Recommends using _extended_shape instead of event_shape
    assert posterior._extended_shape() == diag_posterior._extended_shape()
    assert posterior.batch_shape == diag_posterior.batch_shape


@pytest.mark.parametrize(
    "base_samples, posterior",
    [
        # Try different distibutions
        (torch.ones([1, 1]), posterior_n2_t1()),  # MultivariateNormal minimal example of broadcasting to 2 points
        (torch.ones([1, 2]), posterior_n3_t2()),  # MultitaskMultivariateNormal
        (torch.ones([1, 2]), posterior_b2_n3_t2()),  # MultitaskMultivariateNormal batched
        # try different batch shapes
        (torch.ones([7, 7, 1]), posterior_n2_t1()),  # MultivariateNormal minimal example of broadcasting to 2 points
        (torch.ones([7, 7, 2]), posterior_n3_t2()),  # MultitaskMultivariateNormal
        (torch.ones([7, 7, 2]), posterior_b2_n3_t2()),  # MultitaskMultivariateNormal batched
    ],
)
def test_extend_base_samples_correct_output_shape(base_samples: torch.Tensor, posterior: GPyTorchPosterior):
    expanded_samples = IndependentMCSampler._extend_base_samples(base_samples, posterior)
    assert expanded_samples.shape == base_samples.shape[:-1] + posterior.shape()


def test_extend_base_samples_check_base_sample_sucessfully_duplicated_MultivariateNormal():
    """Make sure the base sample is applied to each x."""
    base_sample = torch.arange(4).reshape(2, 2, 1)
    posterior = posterior_n2_t1()
    expanded_samples = IndependentMCSampler._extend_base_samples(base_sample, posterior)

    # expanded_samples is (samples_shape, *b,n,t).
    # Check each sample has been spread the posterior points in the corresponding index
    assert (expanded_samples[0][0] == base_sample[0][0]).all()
    assert (expanded_samples[0][1] == base_sample[0][1]).all()
    assert (expanded_samples[1][0] == base_sample[1][0]).all()
    assert (expanded_samples[1][1] == base_sample[1][1]).all()


def test_extend_base_samples_check_base_sample_sucessfully_duplicated_MultitaskMultivariateNormal_batched():
    """Make sure the base sample is applied to each x."""
    base_sample = torch.arange(4).reshape(2, 2)
    posterior = posterior_b2_n3_t2()
    expanded_samples = IndependentMCSampler._extend_base_samples(base_sample, posterior)

    # expanded_samples is (samples_shape, *b,n,t).
    # Check each sample has been spread the posterior points in the corresponding index

    # expanded_samples[0]: (2,3,2)  base_sample[0]: (2)
    # check that all the batches b=2, and all points n=3 has the targets = basesample[0]
    assert (expanded_samples[0] == base_sample[0]).all()
    assert (expanded_samples[0] == base_sample[0]).all()


@pytest.mark.integration()
def test_forward_same_samples_used_at_different_n_points_MultivariateNormal():
    r"""Check that the same base sample gets used independantly at each x.

    We override _construct_base_samples' to always reutn the value z = [[1]]. If we then run iwth samples_shape = 1,
    we apply this sample to each of the entries in our posterior. Because independant samplers don't consider the
    covariance between points, we can check each value independantly. Each one should follow this rule:

        $sample = \mu + std * z$   (where z is the base_sample)

    Here we check that using the sampler produce the same result as this.
    """
    manual_base_shape = torch.tensor([[1]])

    class MockIndependentMCSampler(IndependentMCSampler):
        def _construct_base_samples(self, posterior: GPyTorchPosterior) -> None:  # noqa: ARG002
            self.register_buffer("base_samples", manual_base_shape)

    sampler = MockIndependentMCSampler(sample_shape=torch.Size([1]))

    # set up a posterior where it will be be easy to determine if the same samples was used.
    mean = torch.tensor([1, 0.5])
    covar = torch.tensor([[1, 0.1], [0.1, 0.81]])
    posterior = GPyTorchPosterior(MultivariateNormal(mean=mean, covariance_matrix=covar))
    # calucated the independant result for each point
    expect_value_1 = mean[0] + covar[0][0] ** 0.5 * manual_base_shape[0][0]  # should be 1 + 1 = 2
    expect_value_2 = mean[1] + covar[1][1] ** 0.5 * manual_base_shape[0][0]  # should be .5 + .9= 1.4

    # shape (samples_shape = 1,n = 2, m = 1)
    samples = sampler(posterior)

    assert samples[0][0] == expect_value_1
    assert samples[0][1] == expect_value_2


# NOTE: `forward` is checked more extensively in the concerete implementations NoramlIndependentSampler.
