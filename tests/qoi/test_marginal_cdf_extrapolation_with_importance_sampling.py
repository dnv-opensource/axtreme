"""
Tests for MarginalCDFExtrapolation with importance sampling.

Plan:
    - Unit test:
        - _parameter_estimates: input and output weights are attached to the same samples
    - System Test
        - Show that that the QoI estimation with importance samples has less uncertainty in the result compared to when
        using the regular approach of including the whole environment dataset.
"""

# %%
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import torch
from botorch.models.deterministic import GenericDeterministicModel
from botorch.sampling.index_sampler import IndexSampler
from torch.distributions import MultivariateNormal
from torch.utils.data import DataLoader

from axtreme.data import FixedRandomSampler, ImportanceAddedWrapper, MinimalDataset
from axtreme.eval.qoi_job import QoIJob
from axtreme.qoi import MarginalCDFExtrapolation


def test_parameter_estimates_consistency_of_weights(gp_passthrough_1p: GenericDeterministicModel):
    """
    Tests that the `_parameter_estimates` method in `MarginalCDFExtrapolation` correctly combines environment samples
    and their associated importance weights, and that the outputs match expected shapes and values when using a
    deterministic Gaussian Process (GP). The deterministic nature of the GP model leads to fully predictable posterior
    samples, i.e. they are identical to the environment samples.

    This test ensures that:
      - Posterior samples returned by `_parameter_estimates` are correctly shaped and ordered.
      - Importance weights are preserved and correctly matched to the associated samples.

    Args:
        gp_passthrough_1p is defined in conftest.py. It creates a deterministic GP which always produces identical
        posterior samples. The output location is a direct pass through of the given input data, and the scale is set
        to 1e-6.
    """
    # MarginalCDFExtrapolation expects an iterable of env data. To use importance samples, each item is expected to be
    # of the following form [env_samples, importance_weights], where:
    # env_samples.shape = (batch_size,d), and importance_weights.shape = (batch_size,)
    # Note: in practice a dataloader is typically used to achieve this.
    env_and_importance_data = [
        # data batch 1: [env_samples, importance_weights]
        [torch.tensor([[1.0], [2.0]]), torch.Tensor([0.1, 0.2])],
        # data batch 2: [env_samples, importance_weights]
        [torch.tensor([[3.0], [4.0]]), torch.Tensor([0.3, 0.4])],
    ]

    # Run the method
    qoi_estimator = MarginalCDFExtrapolation(
        env_iterable=env_and_importance_data,
        period_len=99999,  # not used in this test
        posterior_sampler=IndexSampler(torch.Size([1])),  # draw 1 posterior sample to simplify comparison
        quantile=torch.tensor(float("nan"), dtype=torch.float64),  # not used in this test
        quantile_accuracy=torch.tensor(float("nan"), dtype=torch.float64),  # not used in this test
        dtype=torch.float64,
    )

    # Get posterior samples and connected weights for given QoI estimator
    posterior_samples, importance_weights_qoi = qoi_estimator._parameter_estimates(gp_passthrough_1p)

    # Verify that the output has the correct dimensions
    # For the posteriors samples the expected shape is: (n_posterior_samples, n_env_samples, n_targets)
    # with n_targets being the number of targets the GP should predict. For gp_passthrough_1p this is set to 2.
    assert posterior_samples.shape == torch.Size([1, 4, 2])
    # For the weights the expected shape is: (n_posterior_samples, n_env_samples)
    assert importance_weights_qoi.shape == torch.Size([1, 4])

    # As a deterministic GP is used the posterior samples should not only have the same values as the
    # input samples but their order should be the same as well
    assert torch.equal(
        torch.cat([batch[0] for batch in env_and_importance_data]).flatten(), posterior_samples[..., 0][0]
    )
    assert torch.equal(
        torch.cat([batch[1] for batch in env_and_importance_data]).flatten(), torch.flatten(importance_weights_qoi)
    )


# Ruff does not allow default arguments in test functions. Using a decorator circumvents that.
@pytest.mark.parametrize("error_tolerance, visualize", [(1, False)])
def test_system_marginal_cdf_with_importance_sampling(
    error_tolerance: float,
    *,
    visualize: bool,
):
    """
    There is variability in the specific samples in the env_data used to instantiate a QoiEstimator, which creates the
    variability of the QoIEstimator estimate. This can be seen by inspecting how the estimates given change when the
    QoiEstimator has been instantiated with a different dataset. This test shows how (good) importance samples can
    reduce the variability between estimates of QoIEstimators instantiated with different env_data. The effect of GP
    uncertainty is removed by using a deterministic model, meaning all uncertainty in the estimate comes from the
    environment samples.

    Args:
        error_tolerance: The error allowed in assertions is multiplied by this number.
        visualize: Bool to specify if the QoI results shall be plotted.

    Expectation:
    As we use a deterministic GP the QoI estimator will not be a distribution but a point representing the mean.
    The std of the means of several runs of the QoI estimator should be lower with uncertainty sampling than without.
    By visual inspection a threshold for the std for importance sampling is chosen.
    """

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
    def _true_underlying_func(x: torch.Tensor) -> torch.Tensor:
        # For this toy example we use a MultivariateNormal distribution

        # Define true loc function
        dist_mean, dist_cov = torch.tensor([1, 1]), torch.tensor([[0.03, 0], [0, 0.03]])
        dist = MultivariateNormal(loc=dist_mean, covariance_matrix=dist_cov)
        loc = torch.exp(dist.log_prob(x))

        # Define true scale function
        scale = torch.ones(x.shape[0]) * 0.1

        return torch.stack([loc, scale], dim=-1)

    gp_deterministic = GenericDeterministicModel(_true_underlying_func, num_outputs=2)

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

    # The aim of this test is to show that QoI estimation with importance sampling results in less uncertainty in the
    # QoI compared to when using the full environment dataset.
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

    if visualize:
        _, ax = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)

        # Plot results for full env data
        df_jobs[df_jobs["dataset_name"] == "full"].hist(column="mean", ax=ax[0], grid=False)
        ax[0].axvline(brute_force_qoi, c="orange", label=f"Brute force ({brute_force_qoi:.2f})")
        ax[0].set_title(
            "Dataset: full, "
            f"mean={df_jobs.loc[df_jobs['dataset_name'] == 'full', 'mean'].mean():.2f}, "
            f"std={df_jobs.loc[df_jobs['dataset_name'] == 'full', 'mean'].std():.2f}"
        )
        ax[0].legend()

        # Plot results for importance sampling
        df_jobs[df_jobs["dataset_name"] == "importance_sample"].hist(column="mean", ax=ax[1], grid=False)
        ax[1].axvline(brute_force_qoi, c="orange", label=f"Brute force ({brute_force_qoi:.2f})")
        ax[1].set_title(
            "Dataset: importance sample, "
            f"mean={df_jobs.loc[df_jobs['dataset_name'] == 'importance_sample', 'mean'].mean():.2f}, "
            f"std={df_jobs.loc[df_jobs['dataset_name'] == 'importance_sample', 'mean'].std():.2f}"
        )
        ax[0].legend()


# %% Can be used to get plots for test_system_marginal_cdf_with_importance_sampling
if __name__ == "__main__":
    # %%
    test_system_marginal_cdf_with_importance_sampling(1, visualize=True)

# %%
