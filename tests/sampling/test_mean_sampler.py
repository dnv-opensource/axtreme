import pytest
import torch
from botorch.posteriors.gpytorch import GPyTorchPosterior

from axtreme.sampling import MeanSampler
from tests.sampling.helpers import (
    posterior_b2_n3_t2,
    posterior_n1_t1,
    posterior_n1_t2,
    posterior_n2_t1,
    posterior_n3_t2,
)


# While this is a simple test, opted to paramterise so that the output dimension is explicit
@pytest.mark.parametrize(
    "posterior,expected_shape",
    [
        (posterior_n1_t1(), torch.Size([1, 1, 1])),
        (posterior_n2_t1(), torch.Size([1, 2, 1])),
        (posterior_n1_t2(), torch.Size([1, 1, 2])),
        (posterior_n3_t2(), torch.Size([1, 3, 2])),
        (posterior_b2_n3_t2(), torch.Size([1, 2, 3, 2])),
    ],
)
def test_forward_takes_one_sample(posterior: GPyTorchPosterior, expected_shape: torch.Size):
    """Uses posterior (parameterised fixture) to test all types of posteriors because expected output is simple."""
    sampler = MeanSampler()
    samples = sampler(posterior)

    # MeanSampler always samples exactly one samples (just collect the mean)
    # Is adds a dimension infront so that it is consisten with the other samplers
    assert samples.shape == expected_shape
