import pytest
import torch
from botorch.models.deterministic import GenericDeterministicModel
from botorch.sampling.index_sampler import IndexSampler
from torch.distributions import Gumbel

from axtreme.qoi.marginal_cdf_extrapolation import MarginalCDFExtrapolation, acceptable_timestep_error, q_to_qtimestep


class TestMarginalCDFExtrapolation:
    """

    Plan:
        - __call__:
            - Unit test None
            - Itegration tests:
                - batcheed params and weights
                - dtype:
                    - float32 where safe
                    - float32 where not safe
        - _parameter_estimates:
            - Integration test:
                - batch produce same results as non batch
                - 1 and multi (2) posterior samples.
                OUT OF SCOPE:
                    - Are importance weights working properly?


    Other sub componets we potentially should test?
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

        Take 2 posterior samples to comfirm the shape can be supported throughout.
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
        # [Gumbel(0, 1e-6), Gumbel(1, 1e-6), Gumbel(2, 1e-6)]. The first two will be clipped to qauntile q= 1-finfo.eps
        #  as per the bounds of ApproximateMixture
        dist = Gumbel(torch.tensor(2, dtype=dtype), 1e-6)
        q_timestep = (dist.cdf(qoi[0]) + (1 - torch.finfo(dtype).eps) * 2) / 3

        # check can be scaled up to the original timeframe with the desired accuracy
        assert q_timestep**period_len == pytest.approx(quantile, abs=quantile_accuracy)

    def test_call_insuffecient_numeric_precision(self, gp_passthrough_1p: GenericDeterministicModel):
        """Runs a minimal version of the qoi using a deterministic model and a short period_len.

        Demonstrate both float32 and float64 can be used (with this short period)

        Take 2 posterior samples to comfirm the shape can be supported throughout.
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
        """Checks that batched and non-batch enve data produce the saem result.

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
    """Numerical stability of converting from period quantiles to timestep quantiles.

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

    This calcutates the "round trip error" of going q_longterm -> q_timestep -> q_longterm, as this is easier to test.
    This error may be larger than conversion q_longterm -> q_timestep.
    By default python uses flaot64 (on most machines). This has a precision of 1e-15.

    NOTE:
        - abs = 1e-10: all tests pass
        - abs = 1e-11: japprox half the tests fail.

    By default python uses flaot64 (on most machines). This has a precision of 1e-15.
    """

    period_len = int(1e13)
    q_step = q_to_qtimestep(q, period_len)
    assert q_step**period_len == pytest.approx(q, abs=1e-3)


def test_acceptable_timestep_error_at_limits_of_precision():
    """When values reach the limits of precision check an error is thrown."""
    with pytest.raises(ValueError):
        _ = acceptable_timestep_error(0.5, int(1e6), atol=1e-10)
