"""
Plan:
    - Unit test:
        - _parameter_estimates: input and output weights are attached to the same samples
    - Integration test:
        - successful run of minimal version of the qoi using a deterministic model with importance sampling
"""

import torch
from botorch.models.deterministic import GenericDeterministicModel
from botorch.sampling.index_sampler import IndexSampler
from torch.utils.data import DataLoader

from axtreme.data import ImportanceAddedWrapper, MinimalDataset
from axtreme.qoi.marginal_cdf_extrapolation import MarginalCDFExtrapolation


def test_parameter_estimates_consistency_of_weights(gp_passthrough_1p: GenericDeterministicModel):
    """
    Run a minimal version of the qoi using a deterministic model and a short period_len to test that the importance
    weights are still connected to the correct samples when using MarginalCDFExtrapolation._parameter_estimates.

    Notes:
    -   gp_passthrough_1p is defined in conftest.py and is a simple deterministic GP that contains 1 unique
        posterior sample
    -   _parameter_estimates needs the importance dataset with specific dimensions by using ImportanceAddedWrapper and
        DataLoader the correct format is ensured and in addition it can be tested that these two functions do not change
        the relation between samples and weights.
    """
    # Define simple importance samples and weights
    importance_samples = torch.Tensor([[1], [2], [3], [4]])
    importance_weights = torch.Tensor([0.1, 0.2, 0.3, 0.4])

    # Wrap them in a dataloader
    importance_dataset = ImportanceAddedWrapper(MinimalDataset(importance_samples), MinimalDataset(importance_weights))
    dataloader = DataLoader(importance_dataset, batch_size=10)

    # Run the method
    qoi_estimator = MarginalCDFExtrapolation(
        env_iterable=dataloader,
        period_len=10,
        posterior_sampler=IndexSampler(torch.Size([1])),  # draw 1 posterior samples to simplify comparison
        quantile=torch.tensor(0.5, dtype=torch.float64),
        quantile_accuracy=torch.tensor(0.1, dtype=torch.float64),
        dtype=torch.float64,
    )

    # Get posterior samples and connected weights for given QoI estimator
    posterior_samples, importance_weights_qoi = qoi_estimator._parameter_estimates(gp_passthrough_1p)

    # As a deterministic GP is used the posterior samples should not only have the same values as the
    # input samples but their order should be the same as well
    assert torch.equal(torch.flatten(importance_samples), posterior_samples[..., 0][0])
    assert torch.equal(torch.flatten(importance_weights), torch.flatten(importance_weights_qoi))


def test_qoi_runs_with_importance_sampling(gp_passthrough_1p: GenericDeterministicModel):
    """
    Run a minimal version of the qoi using a deterministic model and a short period_len to test that it successfully
    runs with importance sampling.

    Notes:
    -   The difference to test_parameter_estimates_consistency_of_weights is that the model is passed directly
        to the QoI estimator not qoi_estimator._parameter_estimates.
    """
    # Define simple importance samples and weights
    importance_samples = torch.Tensor([[1], [2], [3], [4]])
    importance_weights = torch.Tensor([0.1, 0.2, 0.3, 0.4])

    # Wrap them in a dataloader
    importance_dataset = ImportanceAddedWrapper(MinimalDataset(importance_samples), MinimalDataset(importance_weights))
    dataloader = DataLoader(importance_dataset, batch_size=10)

    # Run the method
    qoi_estimator = MarginalCDFExtrapolation(
        env_iterable=dataloader,
        period_len=10,
        posterior_sampler=IndexSampler(torch.Size([1])),  # draw 1 posterior samples to simplify comparison
        quantile=torch.tensor(0.5, dtype=torch.float64),
        quantile_accuracy=torch.tensor(0.1, dtype=torch.float64),
        dtype=torch.float64,
    )

    # This only tests if the functions runs successfully and does not concern itself with the output
    _ = qoi_estimator(gp_passthrough_1p)
