# %%
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

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
    Test overview:
        - without importance sampling:
            - __call__:
                - (unit) multiple posterior samples are handled correctly
                - (unit) marginal CDF with insufficient precision is detected and raises error
                - (integration) Using minimal input compare output to manually calculated expected value
            - _parameter_estimates:
                - (unit) batched and non-batched env data produce the same result
        - with importance sampling:
            - _parameter_estimates:
                - (unit): importance weight w_i is still attached to the correct sample at the end of the function
            - (integration): toy example showing importance sampling can produce less variable QoI estimates.
        - Helper:
            - q_to_qtimestep:
                - (unit) prove that log based calculation are not required for numerical stability
                - (unit) Rough proof results stable for typical values in calculations we do.
            - acceptable_timestep_error:
                - (unit) Error raise when can't perform calculation with required precision
    """

    def test___call__two_posterior_samples(self):
        """__call__ with 2 posterior samples returns a result of shape (2,).

        Mocks _parameter_estimates to return params with n_posterior_samples=2 and
        checks the output shape is (2,), confirming the posterior sample dimension
        propagates correctly through the mixture and icdf steps.
        """
        dtype = torch.float64
        # Shape: (n_posterior_samples=2, n_env=3, n_targets=2)
        params = torch.tensor(
            [[[2.0, 1e-6], [2.0, 1e-6], [2.0, 1e-6]], [[2.0, 1e-6], [2.0, 1e-6], [2.0, 1e-6]]],
            dtype=dtype,
        )
        weights = torch.ones(2, 3, dtype=dtype)  # shape: (n_posterior_samples=2, n_env=3)

        qoi_estimator = MarginalCDFExtrapolation(
            env_iterable=[],  # not used — _parameter_estimates is mocked
            period_len=3,
            quantile=torch.tensor(0.5, dtype=dtype),
            quantile_accuracy=torch.tensor(0.5, dtype=dtype),
            dtype=dtype,
        )

        with patch.object(qoi_estimator, "_parameter_estimates", return_value=(params, weights)):
            qoi = qoi_estimator(MagicMock())

        assert qoi.shape == torch.Size([2])

    def test___call__insufficient_numeric_precision(self):
        """__call__ raises TypeError when float32 precision is too coarse for the period_len.

        With float32 and a realistic 25-year period_len, the single-timestep CDF values
        lack enough resolution for the required quantile_accuracy. The underlying icdf
        solver detects this and raises TypeError. _parameter_estimates is mocked so the
        test exercises only the __call__ precision-check logic.
        """
        dtype = torch.float32
        # Shape: (n_posterior_samples=1, n_env=3, n_targets=2) — simple Gumbel params
        params = torch.tensor([[[2.0, 1e-6], [2.0, 1e-6], [2.0, 1e-6]]], dtype=dtype)
        weights = torch.ones(1, 3, dtype=dtype)

        qoi_estimator = MarginalCDFExtrapolation(
            env_iterable=[],  # not used — _parameter_estimates is mocked
            period_len=25 * 365 * 24,
            quantile=torch.tensor(0.5, dtype=dtype),
            quantile_accuracy=torch.tensor(0.5, dtype=dtype),
            dtype=dtype,
        )

        with (
            patch.object(qoi_estimator, "_parameter_estimates", return_value=(params, weights)),
            pytest.raises(TypeError, match="The distribution provided does not have suitable resolution"),
        ):
            _ = qoi_estimator(MagicMock())

    def test__parameter_estimates_env_batch_invariant(
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

    def test__parameter_estimates_consistency_of_weights(self, gp_passthrough_1p: GenericDeterministicModel):
        """Minor check that the importance weights and samples don't get shuffled during the function.

        In other words, importance weight is attached to the correct sample at the end.

        Approach:
        Use a deterministic GP so the inputs can be traced through to their output position.

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
    def test__call__basic_example(
        self, gp_passthrough_1p: GenericDeterministicModel, dtype: torch.dtype, period_len: int
    ):
        """Minimal MarginalCDFCExtrapolation calculation where expected value can be calculated directly.

        Uses 3 input points and a deterministic GP so expected result can be explicitly calculated.
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
            posterior_sampler=IndexSampler(torch.Size([1])),
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

    @pytest.mark.integration
    @pytest.mark.non_deterministic
    def test_marginal_cdf_with_importance_sampling(  # noqa: PLR0915
        self,
        error_tolerance: float = 1.0,  # noqa: PT028
        *,
        visualise: bool = False,  # noqa: PT028
    ):
        """Toy example showing importance sampling can produce less variable QoI estimates.

        QoIEstimator need to integrate out (marginalise) the effect of the weather. This integration can be ESTIMATED
        using sample based methods. This test demonstrates that importance sampling produces less variable (and still
        correct) estimates compared to random sampling.

        NOTE: QoIEstimator output only shows GP variability. To understand the variability of the QoIEstimator due to
        the environment samples the estimator must me run with different env samples (of the same size).

        Test overview:
            Inputs
            - Response functions: A tight normal distribution at (1,1)
            - Environment distribution: Sample from std normal (with x1 > 0 and x2 > 0)
            - Importance sampling distribution: Uniform samples within a circle of radius 1.75 (with x1 > 0 and x2 > 0)

            Process:
            - 1) draw a subsample from the environment distribution or importance sampling distribution.
            - 2) Run the QoIEstimator with this subsamples (The true underlying function is used inplace of the GP to
              remove the effect of GP uncertainty and isolate the effect of the environment samples). The QoIEstimator
              thus returns a single number per run.
            - 3) Repeat set 1 and 2 many times. Compare the distribution of prediction.

            Expected result:
            - Using the same number of samples, the importance sampling approach should have less variance in its
              its estimates (and still be centered around the true value). Thresholds are set via visual inspection.

        Args:
            error_tolerance: The error allowed in assertions is multiplied by this number.
            visualise: Bool to specify if the QoI results shall be plotted.
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

        if visualise:
            _, ax = plt.subplots(1, 1, figsize=(10, 8))

            # Plot 1: Overlay both distributions on a single scatter plot
            _ = ax.scatter(env_data[:, 0], env_data[:, 1], alpha=0.2, label="Environment", color="steelblue", s=10)
            _ = ax.scatter(
                importance_samples[:, 0].numpy(),
                importance_samples[:, 1].numpy(),
                alpha=0.3,
                label="Importance samples",
                color="darkorange",
                s=10,
            )

            # Plot 2: Heatmap of true underlying function (loc), only where value > 1
            all_x = np.concatenate([env_data[:, 0], importance_samples[:, 0].numpy()])
            all_y = np.concatenate([env_data[:, 1], importance_samples[:, 1].numpy()])
            x1_grid = np.linspace(all_x.min(), all_x.max(), 300)
            x2_grid = np.linspace(all_y.min(), all_y.max(), 300)
            X1, X2 = np.meshgrid(x1_grid, x2_grid)  # noqa: N806
            grid = torch.tensor(np.stack([X1.ravel(), X2.ravel()], axis=1))
            with torch.no_grad():
                loc_values = _true_underlying_func(grid)[:, 0].numpy().reshape(X1.shape)
            masked_values = np.where(loc_values > 1, loc_values, np.nan)
            im = ax.pcolormesh(X1, X2, masked_values, cmap="viridis", shading="auto")
            _ = plt.colorbar(im, ax=ax, label="loc (response)")

            _ = ax.set_title("Environment and importance samples with underlying response function")
            _ = ax.legend

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
            ax[0].set_title(f"Dataset: full, mean={mean_qoi_means_full:.2f}, std={std_qoi_means_full:.2f}")
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
    """Rough check to ensure have enough numerical stability for type of calcs we commonly do.

    This calculates the "round trip error" of going q_longterm -> q_timestep -> q_longterm, as this is easier to test.
    This error may be larger than conversion q_longterm -> q_timestep.
    By default python uses float64 (on most machines). This has a precision of 1e-15.

    Characteristics of common calcs:
    - q = quantile of ERD, typically between 0.05 and 0.95
    - period_len = not expecting to be larger than 100 year simulation with 1 second time step is is approx 1e10.
    - Absolute error typically acceptable in final quantile estimate q +- .001

    NOTE:
        - abs = 1e-10: all tests pass
        - abs = 1e-11: approx half the tests fail.
    """

    period_len = int(1e13)  # every second for 100 year = 1e10
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
    MarginalCDF.test_marginal_cdf_with_importance_sampling(1, visualise=True)
