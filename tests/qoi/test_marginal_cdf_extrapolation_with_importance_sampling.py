"""
Tests for MarginalCDFExtrapolation with importance sampling.

Plan:
    - Unit test:
        - _parameter_estimates: input and output weights are attached to the same samples
    - Integration test:
        - successful run of minimal version of the qoi using a deterministic model with importance sampling
    - System Test
        - Show that that the QoI estimation requires less samples with importance samples compared to when using
        the regular approach of including the whole environment dataset.
"""

# %%
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from botorch.models.deterministic import GenericDeterministicModel
from botorch.sampling.index_sampler import IndexSampler
from numpy.typing import NDArray
from torch.distributions import MultivariateNormal
from torch.utils.data import DataLoader

from axtreme.data import FixedRandomSampler, ImportanceAddedWrapper, MinimalDataset
from axtreme.eval.qoi_job import QoIJob
from axtreme.qoi import MarginalCDFExtrapolation


def test_parameter_estimates_consistency_of_weights(gp_passthrough_1p: GenericDeterministicModel):
    """
    Run a minimal version of the qoi using a deterministic model and a short period_len to test that the importance
    weights are still connected to the correct samples when using MarginalCDFExtrapolation._parameter_estimates.

    Args:
        gp_passthrough_1p is defined in conftest.py. It creates a deterministic GP which always produces identical
        posterior samples. The output location is a direct pass through of the given input data, and the scale is set
        to 1e-6.

    Notes:
    -   _parameter_estimates needs the importance dataset with specific dimensions by using ImportanceAddedWrapper and
        DataLoader the correct format is ensured and in addition it can be tested that these two functions do not change
        the relation between samples and weights.
    """
    # Define simple importance samples and weights
    importance_samples = torch.Tensor([[1], [2], [3], [4]])
    importance_weights = torch.Tensor([0.1, 0.2, 0.3, 0.4])

    # Wrap them together
    importance_dataset = ImportanceAddedWrapper(MinimalDataset(importance_samples), MinimalDataset(importance_weights))

    # The batch size should be smaller than the number of input samples to ensure that the importance_dataset gets
    # split up into smaller batches. By doing this we can test that the batching in _parameter_estimates does not mess
    # up the relation between samples and weights.
    dataloader = DataLoader(importance_dataset, batch_size=2)

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

    Args:
        gp_passthrough_1p is defined in conftest.py. It creates a deterministic GP which always produces identical
        posterior samples. The output location is a direct pass through of the given input data, and the scale is set
        to 1e-6.

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


# These are helper functions to define the loc and scale which are used in the simulator used for the system test
def _true_loc_func(x: NDArray[np.float64]) -> NDArray[np.float64]:
    # For this toy example we use a MultivariateNormal distribution
    dist_mean, dist_cov = torch.tensor([1, 1]), torch.tensor([[0.03, 0], [0, 0.03]])

    dist = MultivariateNormal(loc=dist_mean, covariance_matrix=dist_cov)

    return np.exp(dist.log_prob(torch.tensor(x)).numpy())


def _true_scale_func(x: NDArray[np.float64]) -> NDArray[np.float64]:
    # For this toy example we use a constant scale for simplicity
    return np.ones(x.shape[0]) * 0.1


# Ruff does not allow default arguments in test functions. Using a decorator circumvents that.
@pytest.mark.parametrize("params", [(1, False)])
def test_system_marginal_cdf_with_importance_sampling(
    params: tuple[float, bool],
) -> tuple[pd.DataFrame, float] | None:
    """
    Show that that the QoI estimation requires less samples with importance samples compared to when using
    the regular approach of including the whole environment dataset.

    Args:
        params: tuple of
            - The error allowed in assertions is multiplied by this number.
            - Bool to specify if the QoI results shall be returned for further testing or plotting.

    Returns:
        If return_results==True: the QoI results are returned in a pandas Dataframe

    A deterministic GP is used to remove most of the uncertainty related to the GP. Some uncertainty remains as the GP
    is not fit on the whole dataset.

    Expectation:
    As we use a deterministic GP the QoI estimator will not be a distribution but a point representing the mean.
    The std of the means of several runs of the QoI estimator should be lower with uncertainty sampling than without.
    By visual inspection a threshold for the std for importance sampling is chosen.
    """
    error_tolerance, return_results = params

    # Load precalculated importance samples and weights
    importance_samples = torch.load(
        Path(__file__).parent / "data" / "importance_sampling" / "importance_samples.pt", weights_only=True
    )
    importance_weights = torch.load(
        Path(__file__).parent / "data" / "importance_sampling" / "importance_weights.pt", weights_only=True
    )

    importance_dataset = ImportanceAddedWrapper(MinimalDataset(importance_samples), MinimalDataset(importance_weights))

    # Load environment data
    # The environment is chosen such that it is concentrated in one region of the search space and only few samples
    # cover the subregion where the extreme response occurs
    env_data = np.load(Path(__file__).parent / "data" / "importance_sampling" / "environment_distribution.npy")

    # Get brute force estimate
    brute_force_path = Path(__file__).parent / "data" / "importance_sampling" / "brute_force_solution.json"
    with brute_force_path.open() as file:
        brute_force_qoi = json.load(file)["statistics"]["median"]

    # Set up a deterministic GP which uses the true underlying functions of loc and scale used in the simulator
    def true_underlying_func(x: torch.Tensor) -> torch.Tensor:
        locs = torch.from_numpy(_true_loc_func(x.numpy())).unsqueeze(-1)
        scales = torch.from_numpy(_true_scale_func(x.numpy())).unsqueeze(-1)
        return torch.concat([locs, scales], dim=-1)

    gp_deterministic = GenericDeterministicModel(true_underlying_func, num_outputs=2)

    # Create jobs with with and without importance sampling
    qoi_jobs = []
    datasets = {"full": env_data, "importance_sample": importance_dataset}
    for dataset_name, dataset in datasets.items():
        for i in range(200):
            # A fixed random sampler selects the same samples if the seed is the same which allows the results to be
            # compared if this function is run multiple times
            dataset_size = 800
            sampler = FixedRandomSampler(dataset, num_samples=dataset_size, seed=i, replacement=True)
            dataloader = DataLoader(dataset, sampler=sampler, batch_size=100)

            qoi_estimator = MarginalCDFExtrapolation(
                env_iterable=dataloader,
                period_len=1_000,
                quantile=torch.tensor(0.5),
                quantile_accuracy=torch.tensor(0.01),
                # IndexSampler needs to be used with GenericDeterministicModel. Each sample just selects the mean.
                # As we use a deterministic model all posterior samples are identical and hence we only use one.
                posterior_sampler=IndexSampler(torch.Size([1])),
            )

            qoi_jobs.append(
                QoIJob(
                    name=f"qoi_{dataset_name}_{dataset_size}_{i}",
                    qoi=qoi_estimator,
                    model=gp_deterministic,
                    tags={
                        "dataset_name": dataset_name,
                        "dataset_size": dataset_size,
                    },
                )
            )

    jobs_output_file = None
    qoi_results = [job(output_file=jobs_output_file) for job in qoi_jobs]

    df_jobs = pd.json_normalize([item.to_dict() for item in qoi_results], max_level=1)
    df_jobs.columns = df_jobs.columns.str.removeprefix("tags.")

    # The aim of this test is to show that QoI estimation with importance sampling requires less samples
    # (i.e. num_samples) to converge to the solution than with using the full environment dataset.
    # All absolute values in the following asserts are chosen based on visual inspection of the QoI results.
    # There are two criteria we use to judge the convergence:
    # 1. the mean of the QoI means is close to the brute force solution and in particular the one with
    # importance sampling is closer than the one without
    assert (
        abs(df_jobs.loc[df_jobs["dataset_name"] == "importance_sample", "mean"].mean() - brute_force_qoi)
        <= 0.2 * error_tolerance
    )

    assert abs(df_jobs.loc[df_jobs["dataset_name"] == "importance_sample", "mean"].mean() - brute_force_qoi) <= abs(
        df_jobs.loc[df_jobs["dataset_name"] == "full", "mean"].mean() - brute_force_qoi
    )

    # 2. the std of the QoI means is small with importance sampling
    assert df_jobs.loc[df_jobs["dataset_name"] == "importance_sample", "mean"].std() <= 0.25 * error_tolerance

    if return_results:
        return df_jobs, brute_force_qoi

    return None


# %% Can be used to get plots for test_system_marginal_cdf_with_importance_sampling
if __name__ == "__main__":
    # %% Calculate QoI
    from typing import cast

    qoi_results, brute_force_qoi = cast(
        "tuple[pd.DataFrame, float]", test_system_marginal_cdf_with_importance_sampling((1, True))
    )

    # %% Plot results
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)

    # Plot results for full env data
    qoi_results[qoi_results["dataset_name"] == "full"].hist(column="mean", ax=ax[0], grid=False)
    ax[0].axvline(brute_force_qoi, c="orange", label=f"Brute force ({brute_force_qoi:.2f})")
    ax[0].set_title(
        "Dataset: full, "
        "mean={qoi_results.loc[qoi_results['dataset_name'] == 'full', 'mean'].mean():.2f}, "
        "std={qoi_results.loc[qoi_results['dataset_name'] == 'full', 'mean'].std():.2f}"
    )
    ax[0].legend()

    # Plot results for importance sampling
    qoi_results[qoi_results["dataset_name"] == "importance_sample"].hist(column="mean", ax=ax[1], grid=False)
    ax[1].axvline(brute_force_qoi, c="orange", label=f"Brute force ({brute_force_qoi:.2f})")
    ax[1].set_title(
        "Dataset: importance sample, "
        "mean={qoi_results.loc[qoi_results['dataset_name'] == 'importance_sample', 'mean'].mean():.2f}, "
        "std={qoi_results.loc[qoi_results['dataset_name'] == 'importance_sample', 'mean'].std():.2f}"
    )
    ax[0].legend()

# %%
