import pytest
import torch
from botorch.models.deterministic import GenericDeterministicModel
from botorch.sampling.index_sampler import IndexSampler
from tests.qoi.utils import dummy_posterior_mean


@pytest.fixture
def gp_passthrough_1p() -> GenericDeterministicModel:
    """Simple Gp for testing, contain 1 unique posterior sample.

    Creates a deterministic gp which always produce indentical posterior samples.
    The output loc is a direct pass through of the input data, and the scale is set to 1e-6.

    See dummpy_posterior_mean for details.
    """
    return GenericDeterministicModel(dummy_posterior_mean, num_outputs=2)


@pytest.fixture
def gp_passthrough_1p_sampler() -> IndexSampler:
    """Default sampler to use with gp_passthrough_1p."""
    return IndexSampler(torch.Size([1]))
