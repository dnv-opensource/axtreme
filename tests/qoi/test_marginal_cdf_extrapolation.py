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
from torch.distributions import Gumbel, MultivariateNormal
from torch.utils.data import DataLoader

from axtreme.data import FixedRandomSampler, ImportanceAddedWrapper, MinimalDataset
from axtreme.eval.qoi_job import QoIJob
from axtreme.qoi.marginal_cdf_extrapolation import MarginalCDFExtrapolation, acceptable_timestep_error, q_to_qtimestep

torch.set_default_dtype(torch.float64)


class TestMarginalCDFExtrapolation:
    """

    Plan:
        - without importance sampling:
            - __call__:
                - Unit test None
                - Integration tests:
                    - batched params and weights
                    - dtype:
                        - float32 where safe
                        - float32 where not safe
            - _parameter_estimates:
                - Integration test:
                    - batch produce same results as non batch
                    - 1 and multi (2) posterior samples.
        - with importance sampling:
            - Unit test:
                - _parameter_estimates: input and output weights are attached to the same samples
            - System Test
                - Show that that the QoI estimation with importance samples has less uncertainty in the result compared
                to when using the regular approach of including the whole environment dataset.

    Other sub components we potentially should test?
        - distributions with different batches get optimised/treated properly (e.g in optimisation_)
    """

    @pytest.mark.integration
    @pytest.mark.parametrize(
        "dtype, period_len",
        [
            # Short period
            (torch.float32, 3),
            (torch.float64, 3),
            # Realistic period length
            (torch.float64, 25 * 365 * 24),
        ],
    )
    def test_call_basic_example(
        self, gp_passthrough_1p: GenericDeterministicModel, dtype: torch.dtype, period_len: int
    ):
        """Runs a minimal version of the qoi using a deterministic model and a short period_len.

        Demonstrate both float32 and float64 can be used (with this short period)

        Take 2 posterior samples to confirm the shape can be supported throughout.
        """
        # Define the inputs
        quantile = torch.tensor(0.5, dtype=dtype)
        quantile_accuracy = torch.tensor(0.5, dtype=dtype)
        # fmt: off
        env_sample= torch.tensor(
            [
                [[0], [1], [2]]
            ],
            dtype = dtype
        )
        # fmt: on

        # Run the method
        qoi_estimator_non_batch = MarginalCDFExtrapolation(
            env_iterable=env_sample,
            period_len=period_len,
            posterior_sampler=IndexSampler(torch.Size([2])),  # draw 2 posterior samples
            quantile=quantile,
            quantile_accuracy=quantile_accuracy,
            dtype=dtype,
        )
        qoi = qoi_estimator_non_batch(gp_passthrough_1p)

        # Calculated the expected value directly.
        # This relies on knowledge of the internals, specifically that the underlying distribution produce will be
        # [Gumbel(0, 1e-6), Gumbel(1, 1e-6), Gumbel(2, 1e-6)]. The first two will be clipped to quantile q= 1-finfo.eps
        #  as per the bounds of ApproximateMixture
        dist = Gumbel(torch.tensor(2, dtype=dtype), 1e-6)
        q_timestep = (dist.cdf(qoi[0]) + (1 - torch.finfo(dtype).eps) * 2) / 3

        # check can be scaled up to the original timeframe with the desired accuracy
        assert q_timestep**period_len == pytest.approx(quantile, abs=quantile_accuracy)

    def test_call_insuffecient_numeric_precision(self, gp_passthrough_1p: GenericDeterministicModel):
        """Runs a minimal version of the qoi using a deterministic model and a short period_len.

        Demonstrate both float32 and float64 can be used (with this short period)

        Take 2 posterior samples to confirm the shape can be supported throughout.
        """
        # Define the inputs
        dtype = torch.float32
        quantile = torch.tensor(0.5, dtype=dtype)
        quantile_accuracy = torch.tensor(0.5, dtype=dtype)
        # fmt: off
        env_sample= torch.tensor(
            [
                [[0], [1], [2]]
            ],
            dtype = dtype
        )
        # fmt: on

        # Run the method
        qoi_estimator_non_batch = MarginalCDFExtrapolation(
            env_iterable=env_sample,
            period_len=25 * 365 * 24,
            posterior_sampler=IndexSampler(torch.Size([2])),  # draw 2 posterior samples
            quantile=quantile,
            quantile_accuracy=quantile_accuracy,
            dtype=dtype,
        )

        with pytest.raises(TypeError, match="The distribution provided does not have suitable resolution"):
            _ = qoi_estimator_non_batch(gp_passthrough_1p)

    def test_parameter_estimates_env_batch_invariant(
        self, gp_passthrough_1p_sampler: IndexSampler, gp_passthrough_1p: GenericDeterministicModel
    ):
        """Checks that batched and non-batch env data produce the same result.

        Tests the when the data is split into smaller batches they are then combined into the same output.

        """
        # shape: (6, 1)
        # fmt: off
        env_sample_non_batch = torch.tensor(
            [
                [[0], [1], [2], [3], [4], [5]]
            ]
        )
        # fmt: on

        qoi_estimator_non_batch = MarginalCDFExtrapolation(
            env_iterable=env_sample_non_batch,
            period_len=-1,  # Not used in test
            posterior_sampler=gp_passthrough_1p_sampler,
        )
        param_non_batch, weight_non_batch = qoi_estimator_non_batch._parameter_estimates(gp_passthrough_1p)

        # shape: (2, 3, 1)
        # fmt: off
        env_sample_batch = torch.tensor(
            [
                [[0], [1], [2]],
                [ [3], [4], [5]]
            ]
        )
        # fmt: on
        qoi_estimator_batch = MarginalCDFExtrapolation(
            env_iterable=env_sample_batch,
            period_len=-1,  # Not used in test
            posterior_sampler=gp_passthrough_1p_sampler,
        )
        param_batch, weight_batch = qoi_estimator_batch._parameter_estimates(gp_passthrough_1p)

        torch.testing.assert_close(param_non_batch, param_batch)
        torch.testing.assert_close(weight_non_batch, weight_batch)

    def test_parameter_estimates_consistency_of_weights(self, gp_passthrough_1p: GenericDeterministicModel):
        """
        Tests that the `_parameter_estimates` method in `MarginalCDFExtrapolation` correctly combines environment
        samples and their associated importance weights, and that the outputs match expected values when using a
        deterministic Gaussian Process (GP). The deterministic nature of the GP model leads to fully predictable
        posterior samples, i.e. they are identical to the environment samples.

        This test ensures that:
        - Posterior samples returned by `_parameter_estimates` are correctly ordered.
        - Importance weights are preserved and correctly matched to the associated samples.

        Args:
            gp_passthrough_1p is defined in conftest.py. It creates a deterministic GP which always produces identical
            posterior samples. The output location is a direct pass through of the given input data, and the scale is
            set to 1e-6.
        """
        # MarginalCDFExtrapolation expects an iterable of env data. To use importance samples, each item is expected to
        # be of the following form [env_samples, importance_weights], where:
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

        # As a deterministic GP is used the posterior samples should not only have the same values as the
        # input samples but their order should be the same as well
        # For the deterministic GP gp_passthrough_1p the a dummy posterior mean function is used which sets `loc`
        # equal to the `env_value` and `scale` to 1e-6.
        expected_posterior = torch.tensor(
            [[[1.0000e00, 1.0000e-06], [2.0000e00, 1.0000e-06], [3.0000e00, 1.0000e-06], [4.0000e00, 1.0000e-06]]]
        )
        assert torch.equal(expected_posterior, posterior_samples)

        # The calculated importance weights should be the same as the input importance weights
        assert torch.equal(torch.tensor([[0.1, 0.2, 0.3, 0.4]]), importance_weights_qoi)

    @pytest.mark.system
    @pytest.mark.non_deterministic
    # Ruff does not allow default arguments in test functions. Using a decorator circumvents that.
    @pytest.mark.parametrize("error_tolerance, visualise", [(1, False)])
    def test_system_marginal_cdf_with_importance_sampling(
        self,
        error_tolerance: float,
        *,
        visualise: bool,
    ):
        """
        There is variability in the specific samples in the env_data used to instantiate a QoiEstimator, which creates
        the variability of the QoIEstimator estimate. This can be seen by inspecting how the estimates given change when
        the QoiEstimator has been instantiated with a different dataset. This test shows how (good) importance samples
        can reduce the variability between estimates of QoIEstimators instantiated with different env_data. The effect
        of GP uncertainty is removed by using a deterministic model, meaning all uncertainty in the estimate comes from
        the environment samples.

        Args:
            error_tolerance: The error allowed in assertions is multiplied by this number.
            visualise: Bool to specify if the QoI results shall be plotted.

        Expectation:
        As we use a deterministic GP the QoI estimator will not be a distribution but a point representing the mean.
        The std of the means of several runs of the QoI estimator should be lower with uncertainty sampling than
        without. By visual inspection a threshold for the std for importance sampling is chosen.
        """

        # Load precalculated importance samples and weights
        importance_samples = torch.load(
            Path(__file__).parent / "data" / "importance_sampling" / "importance_samples.pt", weights_only=True
        )
        importance_weights = torch.load(
            Path(__file__).parent / "data" / "importance_sampling" / "importance_weights.pt", weights_only=True
        )

        importance_dataset = ImportanceAddedWrapper(
            MinimalDataset(importance_samples), MinimalDataset(importance_weights)
        )

        # Load environment data
        # The environment is chosen such that it is concentrated in one region of the search space and only few samples
        # cover the subregion where the extreme response occurs
        env_data = np.load(Path(__file__).parent / "data" / "importance_sampling" / "environment_distribution.npy")

        # Get brute force estimate
        brute_force_path = Path(__file__).parent / "data" / "importance_sampling" / "brute_force_solution.json"
        with brute_force_path.open() as file:
            brute_force_qoi = json.load(file)["statistics"]["median"]

        # Set up a deterministic GP
        def _true_underlying_func(x: torch.Tensor) -> torch.Tensor:
            # Define loc function
            dist_mean, dist_cov = torch.tensor([1, 1]), torch.tensor([[0.03, 0], [0, 0.03]])
            dist = MultivariateNormal(loc=dist_mean, covariance_matrix=dist_cov)
            loc = torch.exp(dist.log_prob(x))

            # Define scale function
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

        # The aim of this test is to show that QoI estimation with importance sampling results in less uncertainty in
        # the QoI compared to when using the full environment dataset.

        # All absolute values in the following asserts are chosen based on visual inspection of the QoI results.

        # There are two criteria we use to judge the convergence:

        # 1. By using importance sampling the variance between multiple runs is significantly reduced
        # compared to using the whole environment dataset. This is the main benefit of using importance sampling.
        # We verify this by checking that the std of the QoI means is small with importance sampling.
        std_qoi_means_importance_sampling = df_jobs.loc[df_jobs["dataset_name"] == "importance_sample", "mean"].std()
        assert std_qoi_means_importance_sampling <= 0.25 * error_tolerance

        # 2. The mean of the QoI means is close to the brute force solution.
        mean_qoi_means_importance_sampling = df_jobs.loc[df_jobs["dataset_name"] == "importance_sample", "mean"].mean()
        assert abs(mean_qoi_means_importance_sampling - brute_force_qoi) <= 0.2 * error_tolerance

        if visualise:
            _, ax = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)

            # Plot results for full env data
            std_qoi_means_full = df_jobs.loc[df_jobs["dataset_name"] == "full", "mean"].std()
            mean_qoi_means_full = df_jobs.loc[df_jobs["dataset_name"] == "full", "mean"].mean()

            df_jobs[df_jobs["dataset_name"] == "full"].hist(column="mean", ax=ax[0], grid=False)
            ax[0].axvline(brute_force_qoi, c="orange", label=f"Brute force ({brute_force_qoi:.2f})")
            ax[0].set_title(f"Dataset: full, mean={std_qoi_means_full:.2f}, std={mean_qoi_means_full:.2f}")
            ax[0].legend()

            # Plot results for importance sampling
            df_jobs[df_jobs["dataset_name"] == "importance_sample"].hist(column="mean", ax=ax[1], grid=False)
            ax[1].axvline(brute_force_qoi, c="orange", label=f"Brute force ({brute_force_qoi:.2f})")
            ax[1].set_title(
                "Dataset: importance sample, "
                f"mean={mean_qoi_means_importance_sampling:.2f}, "
                f"std={std_qoi_means_importance_sampling:.2f}"
            )
            ax[0].legend()


@pytest.mark.parametrize("dtype", [(torch.float32), (torch.float64)])
@pytest.mark.parametrize(
    "longterm_q, period_len",
    [
        # One hour simulation for 50 years
        (0.5, int(50 * 365.25 * 24 * 1)),
        # higher perentile
        (0.9, int(50 * 365.25 * 24 * 1)),
        # 10 minute simulation for 50 years
        (0.5, int(50 * 365.25 * 24 * 6)),
    ],
)
def test_q_to_qtimestep_numerical_precision_of_timestep_conversion(
    dtype: torch.dtype, longterm_q: float, period_len: int
):
    """Numerical stability of converting from period quantiles to time step quantiles.

    This serves as documentation to show standard operators do not cause numerical issues when converting.
    """

    q = torch.tensor(longterm_q, dtype=dtype)
    # simple caclution
    _q_step = q_to_qtimestep(q.item(), period_len)
    q_step = torch.tensor(_q_step, dtype=dtype)

    # log based
    q_step_exp = torch.exp(torch.log(q) / period_len)

    # This is approx only for float 32, as q_to_qtimestep internally operates in float64.
    # eps is smallest representable step from 1-2, from .5 to 1 get extra bit of precision
    torch.testing.assert_close(q_step, q_step_exp)


@pytest.mark.parametrize(
    "q", [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
)
def test_q_to_qtimestep_numerical_precision_period_increase(q: float):
    """Estimate the numerical error introduced through this operation.

    This calculates the "round trip error" of going q_longterm -> q_timestep -> q_longterm, as this is easier to test.
    This error may be larger than conversion q_longterm -> q_timestep.
    By default python uses float64 (on most machines). This has a precision of 1e-15.

    NOTE:
        - abs = 1e-10: all tests pass
        - abs = 1e-11: approx half the tests fail.

    By default python uses float64 (on most machines). This has a precision of 1e-15.
    """

    period_len = int(1e13)
    q_step = q_to_qtimestep(q, period_len)
    assert q_step**period_len == pytest.approx(q, abs=1e-3)


def test_acceptable_timestep_error_at_limits_of_precision():
    """When values reach the limits of precision check an error is thrown."""
    with pytest.raises(ValueError):
        _ = acceptable_timestep_error(0.5, int(1e6), atol=1e-10)


# %% Can be used to get plots for test_system_marginal_cdf_with_importance_sampling
if __name__ == "__main__":
    # %%
    MarginalCDF = TestMarginalCDFExtrapolation()
    MarginalCDF.test_system_marginal_cdf_with_importance_sampling(1, visualise=True)
