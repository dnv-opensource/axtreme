# %%
from unittest.mock import MagicMock, patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import torch
from botorch.models.deterministic import GenericDeterministicModel
from botorch.sampling.index_sampler import IndexSampler
from setuptools import Distribution
from torch.distributions import Categorical, Gumbel, MultivariateNormal, Normal, Uniform
from torch.utils.data import DataLoader, RandomSampler, TensorDataset

from axtreme.data import FixedRandomSampler, ImportanceAddedWrapper, MinimalDataset
from axtreme.distributions.mixture import ApproximateMixture
from axtreme.eval.qoi_job import QoIJob
from axtreme.qoi.marginal_cdf_extrapolation import (
    MarginalCDFExtrapolation,
    _create_lowerbound_degenerate_distribution_params,
    _mixture_distribution_from_importance_samples,
    acceptable_timestep_error,
    q_to_qtimestep,
)
from axtreme.utils.brute_force import brute_force_calc

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
    def test_marginal_cdf_with_importance_sampling(  # noqa: C901, PLR0915
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
        ######### STEP UP
        PERIOD_LEN = 1_000  # noqa: N806
        N_BRUTE_FORCE_ESTS = 20_00  # noqa: N806

        ### Make and sample env dist
        class _EnvDist:
            """Simple truncated normal distribution"""

            def __init__(self, loc: torch.Tensor, cov: torch.Tensor, lower_bounds: torch.Tensor) -> None:
                assert cov.shape == (2, 2), "NotYetImplemented, only supports 2D case currently"
                assert (cov[0, 1] == 0) and (cov[1, 0] == 0), "NotYetImplemented, only supports independent normal"  # noqa: PT018
                self.mvn = MultivariateNormal(loc, covariance_matrix=cov)
                self.lower_bounds = lower_bounds
                # Calculated how much of mass has been removed by truncation.
                # Pdf will need to increase so the integral is 1.
                prb_not_truncated = (1 - Normal(loc[0], cov[0, 0] ** 0.5).cdf(lower_bounds[0])) * (
                    1 - Normal(loc[1], cov[1, 1] ** 0.5).cdf(lower_bounds[1])
                )
                self._pdf_correction_factor = 1 / prb_not_truncated

            def sample(self, sample_shape: torch.Size) -> torch.Tensor:
                """Sample shape restricted to (n,)"""
                samples = torch.tensor([]).reshape(0, 2)
                while samples.shape[0] < sample_shape[0]:
                    s = self.mvn.sample(sample_shape)
                    in_region = (s > self.lower_bounds).all(dim=-1)
                    samples = torch.cat([samples, s[in_region]], dim=0)

                return samples[: sample_shape[0]]

            def pdf(self, x: torch.Tensor) -> torch.Tensor:
                """PDF of the truncated normal distribution."""
                in_region = (x > self.lower_bounds).all(dim=-1)
                pdf = torch.zeros(x.shape[0])
                pdf[in_region] = self.mvn.log_prob(x[in_region]).exp() * self._pdf_correction_factor
                return pdf

        env_dist = _EnvDist(
            loc=torch.tensor([0.1, 0.1]), cov=torch.tensor([[0.2, 0], [0, 0.2]]), lower_bounds=torch.tensor([0.0, 0.0])
        )
        with torch.random.fork_rng():
            _ = torch.manual_seed(66)
            env_samples = env_dist.sample(torch.Size([PERIOD_LEN * N_BRUTE_FORCE_ESTS]))
        env_dataset = MinimalDataset(env_samples)

        ### Importance sampling distribution: Uniform over region 0 < x1 < 2, 0 < x2 < 2
        is_dist_bounds = torch.tensor([[0.0, 0.0], [2.0, 2.0]])
        with torch.random.fork_rng():
            _ = torch.manual_seed(66)
            importance_samples = Uniform(0, 2).sample(torch.Size([10_000, 2]))

        def importance_pdf(x: torch.Tensor) -> torch.Tensor:
            in_region = (
                (x[:, 0] >= is_dist_bounds[0, 0])
                & (x[:, 0] <= is_dist_bounds[1, 0])
                & (x[:, 1] >= is_dist_bounds[0, 1])
                & (x[:, 1] <= is_dist_bounds[1, 1])
            )
            pdf = torch.zeros(x.shape[0])
            area = (is_dist_bounds[1, 0] - is_dist_bounds[0, 0]) * (is_dist_bounds[1, 1] - is_dist_bounds[0, 1])
            pdf[in_region] = 1 / area
            return pdf

        importance_weights = env_dist.pdf(importance_samples) / importance_pdf(importance_samples)
        importance_dataset = ImportanceAddedWrapper(
            MinimalDataset(importance_samples), MinimalDataset(importance_weights)
        )

        ### Define the response functions
        def _true_underlying_func(x: torch.Tensor) -> torch.Tensor:
            # Define loc function
            dist_mean, dist_cov = torch.tensor([1, 1]), torch.tensor([[0.03, 0], [0, 0.03]])
            dist = MultivariateNormal(loc=dist_mean, covariance_matrix=dist_cov)
            loc = torch.exp(dist.log_prob(x))

            # Define scale function
            scale = torch.ones(x.shape[0]) * 0.1

            return torch.stack([loc, scale], dim=-1)

        ## Calculate the brute force solutions
        generator = torch.Generator()
        _ = generator.manual_seed(27)
        response_samples, xmax_samples = brute_force_calc(
            dataloader=DataLoader(
                TensorDataset(env_samples),
                batch_size=1000,
                sampler=RandomSampler(env_dataset, num_samples=PERIOD_LEN, replacement=False, generator=generator),
            ),
            response_params_func=_true_underlying_func,
            response_dist_class=Gumbel,
            num_estimates=N_BRUTE_FORCE_ESTS,
        )
        assert xmax_samples.unique(dim=0).shape[0] / N_BRUTE_FORCE_ESTS > 0.75, (
            "Duplicate xmax indicates insufficient variety of env_samples, which can create unstable estimates."
        )
        brute_force_qoi = torch.median(response_samples)
        # TODO(sw 2026-04-20): Add an uncertainty est to brute force est (so know have enough samples).

        if visualise:
            _, ax = plt.subplots(1, 1, figsize=(10, 8))

            # Plot 1: Overlay both distributions on a single scatter plot
            _ = ax.scatter(
                env_samples[:, 0], env_samples[:, 1], alpha=0.2, label="Environment", color="steelblue", s=10
            )
            _ = ax.scatter(
                importance_samples[:, 0].numpy(),
                importance_samples[:, 1].numpy(),
                alpha=0.3,
                label="Importance samples",
                color="darkorange",
                s=10,
            )

            # Plot 2: Heatmap of true underlying function (loc), only where value > 1
            all_x = np.concatenate([env_samples[:, 0].numpy(), importance_samples[:, 0].numpy()])
            all_y = np.concatenate([env_samples[:, 1].numpy(), importance_samples[:, 1].numpy()])
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

        ####### Run the estimations
        gp_deterministic = GenericDeterministicModel(_true_underlying_func, num_outputs=2)

        # Create jobs with with and without importance sampling
        qoi_jobs = []
        datasets = {"full": env_dataset, "importance_sample": importance_dataset}
        for dataset_name, dataset in datasets.items():
            for i in range(200):
                # A fixed random sampler selects the same samples if the seed is the same which allows the results to be
                # compared if this function is run multiple times
                dataset_size = 800
                sampler = FixedRandomSampler(
                    dataset, num_samples=dataset_size, seed=i, replacement=True
                )  # typing: ignore[arg-type]
                dataloader: DataLoader[tuple[torch.Tensor, ...]] = DataLoader(dataset, sampler=sampler, batch_size=100)

                qoi_estimator = MarginalCDFExtrapolation(
                    env_iterable=dataloader,
                    period_len=PERIOD_LEN,
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


class TestMixtureDistributionFromImportanceSamples:
    def test_unbatched_input(self):
        dtype = torch.float64
        weights = torch.ones(4, dtype=dtype) / 4
        loc = torch.tensor([0.0, 10.0, 20.0, 30.0], dtype=dtype)
        scale = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=dtype)
        params = torch.concat([loc.unsqueeze(-1), scale.unsqueeze(-1)], dim=-1)

        dist, _ = _mixture_distribution_from_importance_samples(weights, params, Gumbel)

        # expected a new distribution with weight .75 to be added at the end
        expected_weights = torch.tensor([0.0625, 0.0625, 0.0625, 0.0625, 0.75], dtype=dtype)
        torch.testing.assert_close(dist.mixture_distribution.probs, expected_weights)

        # expect the component distribution size to be larger.
        # Checking the additional value is appropriate is left to _create_lowerbound_degenerate_distribution_params
        assert dist.component_distribution.loc.shape == torch.Size([5])

    def test_basic_batched_input(self):
        """Test batched inputs (e.g. params (*b, n_samples, n_targets)) are handled and each batch gets a unique
        additional component"""
        dtype = torch.float64
        weights = torch.ones(2, 4, dtype=dtype)
        weights[0] /= 4
        weights[1] /= 5
        loc = torch.tensor([[0.0, 10.0, 20.0, 30.0], [40.0, 50.0, 60.0, 70.0]], dtype=dtype)
        scale = torch.ones_like(loc)
        params = torch.concat([loc.unsqueeze(-1), scale.unsqueeze(-1)], dim=-1)

        dist, _ = _mixture_distribution_from_importance_samples(weights, params, Gumbel)

        expected_weights = torch.tensor(
            [[0.0625, 0.0625, 0.0625, 0.0625, 0.75], [0.05, 0.05, 0.05, 0.05, 0.8]], dtype=dtype
        )
        torch.testing.assert_close(dist.mixture_distribution.probs, expected_weights)

        # Checking the add params finding adding cdf contribution before the component dist is left to
        # TestCreateLowerboundDegenerateDistributionParams. Here small check that each batch got a unique addition.
        assert dist.component_distribution.loc.shape == torch.Size([2, 5])
        # Rough sanity check that additional components are added to right place
        assert dist.component_distribution.loc[0, -1].item() == pytest.approx(0 - 3, abs=3)
        assert dist.component_distribution.loc[1, -1].item() == pytest.approx(40 - 3, abs=3)

    @pytest.mark.parametrize(
        "p_dist, q_dist, expected_result",
        [
            # Case supp(q) < supp(p).
            (Normal(0, 1), Uniform(0, 3), 0.5),
            # Case supp(q) >= supp(p).
            (Uniform(0, 3), Normal(0, 1), 1.0),
        ],
    )
    def test_p_over_q_integral_estimate_and_uncertainty(
        self, p_dist: Distribution, q_dist: Distribution, expected_result: float
    ):
        """Ignore the component distribution and check the integral int_supp(q){(p)} has good mean.

        Perform a calculation where the result is know analytically.
        """
        # importance samples from q
        with torch.random.fork_rng():
            _ = torch.manual_seed(10)
            # Take a large number of sample as currently only interested in checking the integral estimate is good.b
            sample = q_dist.sample((10_000,))  # pyright: ignore[reportAttributeAccessIssue]

        # Calculate the weights. This gets a bit convoluted because we need to manually set values outside the support.
        in_p_support = p_dist.support.check(sample)  # pyright: ignore[reportAttributeAccessIssue]
        weights = torch.zeros_like(sample)
        weights[in_p_support] = torch.exp(p_dist.log_prob(sample[in_p_support]) - q_dist.log_prob(sample[in_p_support]))  # pyright: ignore[reportAttributeAccessIssue]

        dummy_params = torch.ones((*sample.shape, 2))

        dist, _ = _mixture_distribution_from_importance_samples(
            weights=weights,
            params=dummy_params,
            component_dist_class=Gumbel,  # Note this is just a placeholder, we don't care about the component dist
        )

        # integral_p_over_not_q is added to the end of the weights
        integral_p_over_q_est = 1 - dist.mixture_distribution.probs[-1]
        assert integral_p_over_q_est.item() == pytest.approx(expected_result, abs=0.01)

    # TODO(sw 2026-04-12): could add test to check uncertainty is well calibrated. Skipping for now as the formula
    # is pretty straight forward and we don't currently rely on the number.

    def test_p_over_q_integral_estimate_clipping_and_warning(self):
        """Simple test to check weight clipping and warning when int_supp(q)(p) > 1."""
        dtype = torch.float64
        n_samples = 4
        # weights with mean = 2.0 > 1, simulating an over-estimated integral
        weights = torch.full((n_samples,), 2.0, dtype=dtype)
        loc = torch.tensor([0.0, 10.0, 20.0, 30.0], dtype=dtype)
        scale = torch.ones(n_samples, dtype=dtype)
        params = torch.concat([loc.unsqueeze(-1), scale.unsqueeze(-1)], dim=-1)

        with pytest.warns(UserWarning, match="exceeds 1"):
            dist, _ = _mixture_distribution_from_importance_samples(weights, params, Gumbel)

        # When clipped: integral_p_over_not_q = 1 - 1 = 0, so the degenerate component has weight 0
        assert dist.mixture_distribution.probs[-1].item() == pytest.approx(0.0, abs=1e-10)

    def test_cdf_equivalent_when_q_subset_q(self, visualise: bool = False):  # noqa: FBT001, FBT002, PT028
        """Tests importance sampling can exclude regions of 0 response (corresponding to CDF(R<=0) = 1).

        ``_mixture_distribution_from_importance_samples`` is designed to relax importance sampling requirements:
        - from: q(x) = 0 -> f(x)p(x) = 0
        - to: q(x) = 0 -> f(x) = 1 OR p(x) = 0
        This test checks the CDF produced when q excludes regions of 0 response is correct.

        Inputs:
        - Environment distributions: Uniform(0,1)
        - Importance sampling distribution: Uniform(0.75,1)
        - Response function: The conceptually show how the response is generated (the response function is not used in
          the test)
            - x in [0,.75]: 0
            - x in [.75,1]: Gumbel(loc = 1, scale = 1)
        - true_underlying function: Generates the location and scale corresponding to the response function
            - x in [0,.75] -> loc = 0, scale = 1e-6 (to approximate a step function)
            - x in [.75,1] -> loc = 1, scale = 1

        Process:
        - 1) calculate the resulting CDF without importance sampling (or analytically)
        - 2) calculate the CDF with importance sampling (where q excludes regions of 0 response)

        Expected result:
            Both produce the same CDF
        """
        env_dist = Uniform(0, 1)
        is_dist = Uniform(0.75, 1)

        analytical_dist = ApproximateMixture(
            mixture_distribution=Categorical(torch.tensor([0.75, 0.25])),  # 75% mass on zero response, 25% on active
            component_distribution=Gumbel(loc=torch.tensor([0.0, 1.0]), scale=torch.tensor([1e-6, 1.0])),
        )

        def true_underlying(x: torch.Tensor) -> torch.Tensor:
            in_active_region = x >= 0.75
            loc = torch.where(in_active_region, torch.tensor(1.0), torch.tensor(0.0))
            scale = torch.where(in_active_region, torch.tensor(1.0), torch.tensor(1e-6))
            return torch.stack([loc, scale], dim=-1)

        # Importance sample (q subset p):
        n_samples = 10_000
        with torch.random.fork_rng():
            _ = torch.manual_seed(10)
            is_samples = is_dist.sample(torch.Size((n_samples,)))
        # p(x)/q(x) = 1/0.25 = 4
        is_weights = torch.exp(env_dist.log_prob(is_samples) - is_dist.log_prob(is_samples))
        is_params = true_underlying(is_samples)
        # NOTE: use arguement `lower_bound`` if want to plot over the entire domain.
        # Typically our optimisation only searches in the importance region as we look for large q value
        marginal_dist, _ = _mixture_distribution_from_importance_samples(
            weights=is_weights, params=is_params, component_dist_class=Gumbel
        )

        # Check the distribution the approximately equal (within the important bounds)
        x_range = torch.linspace(0.75, 1, 100)
        expected_cdf = analytical_dist.cdf(x_range)
        actual_cdf = marginal_dist.cdf(x_range)
        torch.testing.assert_close(expected_cdf, actual_cdf, atol=0.01, rtol=0.0)

        if visualise:
            # for demo purposes can also put the MC estimate on the plot:
            # Direct monte carlo estimate:
            n_samples = 10_000
            env_samples = env_dist.sample(torch.Size((n_samples,)))
            params = true_underlying(env_samples)
            # Not importance sampling, so no weights
            weights = torch.ones_like(env_samples)
            mc_dist = ApproximateMixture(
                mixture_distribution=Categorical(weights),
                component_distribution=Gumbel(loc=params[:, 0], scale=params[:, 1]),
            )

            _, ax = plt.subplots(1, 1, figsize=(10, 8))
            _ = ax.plot(x_range, mc_dist.cdf(x_range), label="MC CDF", color="green", linestyle="dotted")
            _ = ax.plot(x_range, expected_cdf, label="Analytical CDF", color="blue")
            _ = ax.plot(x_range, actual_cdf, label="IS CDF", color="orange", linestyle="dashed")
            _ = ax.set_title("CDF comparison between analytical and importance sampling approach")
            _ = ax.set_xlabel("x")
            _ = ax.set_ylabel("CDF")
            _ = ax.legend()


class TestCreateLowerboundDegenerateDistributionParams:
    def test_simple_dist(self):
        """Degenerate dist finishes adding mass before the single component starts."""
        comp_dist = Gumbel(loc=torch.tensor([0.0, 10.0]), scale=torch.tensor([1.0, 1.0]))
        params = _create_lowerbound_degenerate_distribution_params(comp_dist)
        dtype = params.dtype
        finfo = torch.finfo(dtype)
        # Every component's lower tail must start at or above where the degenerate dist finishes
        component_lower_bounds = comp_dist.icdf(torch.tensor(finfo.eps)).min(dim=-1).values  # noqa: PD011
        degenerate_dist_upper_bound = Gumbel(*torch.unbind(params, dim=-1)).icdf(torch.tensor(1 - finfo.eps))
        assert component_lower_bounds >= degenerate_dist_upper_bound

    def test_batched_dist_different_scales(self):
        """Degenerate dist is placed before the lowest-starting component when locs and scales differ."""
        comp_dist = Gumbel(loc=torch.tensor([[0.0, 10.0], [1.0, 1.0]]), scale=torch.tensor([[0.5, 5.0], [1.0, 1.0]]))
        params = _create_lowerbound_degenerate_distribution_params(comp_dist)
        dtype = params.dtype
        finfo = torch.finfo(dtype)
        # The degenerate dist must finish before ALL components start adding mass
        component_lower_bounds = comp_dist.icdf(torch.tensor(finfo.eps)).min(dim=-1).values  # noqa: PD011
        degenerate_dist_upper_bound = Gumbel(*torch.unbind(params, dim=-1)).icdf(torch.tensor(1 - finfo.eps))
        assert (component_lower_bounds >= degenerate_dist_upper_bound).all()

    def test_lower_bound_too_high_raises(self):
        """Passing a lower_bound above the component dist's lower tail raises ValueError."""
        comp_dist = Gumbel(loc=torch.tensor([0.0, 10.0]), scale=torch.tensor([1.0, 1.0]))
        dtype = comp_dist.loc.dtype
        finfo = torch.finfo(dtype)
        actual_lower = comp_dist.icdf(torch.tensor(finfo.eps)).min().item()
        too_high = actual_lower + 1.0
        with pytest.raises(ValueError, match="lower_bound provided"):
            _ = _create_lowerbound_degenerate_distribution_params(comp_dist, lower_bound=too_high)

    def test_lower_bound_set_appropriately(self):
        """When lower_bound is set below all components, degenerate dist finishes at or before lower_bound."""
        comp_dist = Gumbel(loc=torch.tensor([0.0, 10.0]), scale=torch.tensor([1.0, 1.0]))
        lower_bound = -20.0
        params = _create_lowerbound_degenerate_distribution_params(comp_dist, lower_bound=lower_bound)
        dtype = params.dtype
        finfo = torch.finfo(dtype)
        # The degenerate dist must have finished adding mass by lower_bound
        assert Gumbel(*torch.unbind(params, dim=-1)).icdf(torch.tensor(1 - finfo.eps)).item() <= lower_bound
