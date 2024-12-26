import re
from pathlib import Path
from unittest.mock import patch

import matplotlib.pyplot as plt
import pytest
import torch
from torch.distributions import Categorical, Distribution, Gumbel, Normal

from axtreme.distributions.icdf import icdf, icdf_1d
from axtreme.distributions.mixture import ApproximateMixture
from axtreme.qoi.marginal_cdf_extrapolation import acceptable_timestep_error, q_to_qtimestep

# We need to work to high prcision to avoid numerical issues.
# TODO(sw 2024-12-11): This should eventually be moved to the specific tests as their precision requirements are known
# cdfs if float32 can give very different results to float64 in the extreme. as we often check by converting back
# through the cdf. Do not assume because because nothing fails when running float32 that there are no issues.
torch.set_default_dtype(torch.float64)


class TestICDF:
    """These the orchestration of icdf_1d. We want to test:
    - The batch (which gets flattened) ends up with results in the right place
    - the quantiles are expanded correctly
    - the bounds are expanded correctly

    TODO(sw 2024-12-12): These tests feel a little bit hacky, is there a nice way to clean them up?
    """

    def test_expanded_inputs_produce_expected_results(self):
        """Check output locations are correct when all inputs are provided in the expanded shape.

        In this test we patch the icdf_1f that is called internally, as we are only testing the orchestrartion of this
        function (e.g input and outputs all go to the right place). The method for doing this is a bit hacky.
        """

        def mock_cdf(
            dist: Distribution,
            quantile: float,
            max_acceptable_error: float,  # noqa: ARG001
            bounds: tuple[float, float],
        ) -> torch.Tensor:
            """Encodes the inputs into a single tensor which can be tracked though the function.

            This is a bit hack because we can't make a seperate element for each input becuase of the reshaping that
            occurs internally. Instead we use the different digits of a number to represent the different inputs.
            This is obviously fragile.


            Return:
                A tensor = loc + quantile + bounds[0] * 0.01 + bounds[1] * 0.001
            """
            result = dist.loc + quantile + bounds[0] * 0.01 + bounds[1] * 0.001  # type: ignore  # noqa: PGH003
            return torch.tensor([result])

        loc = torch.tensor([[1.0, 2, 3], [4, 5, 6]])
        scale = torch.ones_like(loc)
        # Batch shape(2,3)
        dist = Gumbel(loc, scale)

        # shape: 2,3
        quantile = loc * 0.1
        max_acceptable_error = 1e-5
        # shape (2,2, 3)
        bounds = torch.tensor([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])

        with patch("axtreme.distributions.icdf.icdf_1d", new=mock_cdf):
            result = icdf(dist, quantile, max_acceptable_error, bounds)

        assert result.shape == (2, 3)

        # same asL loc + quantile + bounds[0] * 0.01 + bounds[1] * 0.001
        expected_results = torch.tensor([[1.1110, 2.2220, 3.3330], [4.4440, 5.5550, 6.6660]])
        torch.testing.assert_close(result, expected_results)

    @pytest.mark.parametrize(
        "quantile, expected_results",
        [
            (torch.tensor(0.9), torch.tensor([[1.9110, 2.9220, 3.9330], [4.9440, 5.9550, 6.9660]])),
            (torch.tensor([[0.8], [0.9]]), torch.tensor([[1.8110, 2.8220, 3.8330], [4.9440, 5.9550, 6.9660]])),
        ],
    )
    def test_q_shapes(self, quantile: torch.Tensor, expected_results: torch.Tensor):
        """Duplicate of the above, but with different shapes of q. This is to test that the q is expanded correctly."""

        def mock_cdf(
            dist: Distribution,
            quantile: float,
            max_acceptable_error: float,  # noqa: ARG001
            bounds: tuple[float, float],
        ) -> torch.Tensor:
            """Encodes the inputs into a single tensor which can be tracked though the function.

            This is a bit hack because we can't make a seperate element for each input becuase of the reshaping that
            occurs internally. Instead we use the different digits of a number to represent the different inputs.
            This is obviously fragile.


            Return:
                A tensor = loc + quantile + bounds[0] * 0.01 + bounds[1] * 0.001
            """
            result = dist.loc + quantile + bounds[0] * 0.01 + bounds[1] * 0.001  # type: ignore  # noqa: PGH003
            return torch.tensor([result])

        loc = torch.tensor([[1.0, 2, 3], [4, 5, 6]])
        scale = torch.ones_like(loc)
        # Batch shape(2,3)
        dist = Gumbel(loc, scale)

        max_acceptable_error = 1e-5
        # shape (2,2, 3)
        bounds = torch.tensor([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])

        with patch("axtreme.distributions.icdf.icdf_1d", new=mock_cdf):
            result = icdf(dist, quantile, max_acceptable_error, bounds)

        # same asL loc + quantile + bounds[0] * 0.01 + bounds[1] * 0.001
        torch.testing.assert_close(result, expected_results)

    @pytest.mark.parametrize(
        "bounds, expected_results",
        [
            (torch.tensor([8, 9]).reshape(2, 1, 1), torch.tensor([[1.1890, 2.2890, 3.3890], [4.4890, 5.5890, 6.6890]])),
            (
                torch.tensor([[6, 8], [7, 9]]).reshape(2, 2, 1),
                torch.tensor([[1.1670, 2.2670, 3.3670], [4.4890, 5.5890, 6.6890]]),
            ),
        ],
    )
    def test_bounds_shapes(self, bounds: torch.Tensor, expected_results: torch.Tensor):
        """Duplicate of the above, but with different shapes of bounds. This is to test that the bounds are expanded
        correctly.
        """

        def mock_cdf(
            dist: Distribution,
            quantile: float,
            max_acceptable_error: float,  # noqa: ARG001
            bounds: tuple[float, float],
        ) -> torch.Tensor:
            """Encodes the inputs into a single tensor which can be tracked though the function.

            This is a bit hack because we can't make a seperate element for each input becuase of the reshaping that
            occurs internally. Instead we use the different digits of a number to represent the different inputs.
            This is obviously fragile.


            Return:
                A tensor = loc + quantile + bounds[0] * 0.01 + bounds[1] * 0.001
            """
            result = dist.loc + quantile + bounds[0] * 0.01 + bounds[1] * 0.001  # type: ignore  # noqa: PGH003
            return torch.tensor([result])

        loc = torch.tensor([[1.0, 2, 3], [4, 5, 6]])
        scale = torch.ones_like(loc)
        # Batch shape(2,3)
        dist = Gumbel(loc, scale)
        quantile = loc * 0.1
        max_acceptable_error = 1e-5

        with patch("axtreme.distributions.icdf.icdf_1d", new=mock_cdf):
            result = icdf(dist, quantile, max_acceptable_error, bounds)

        # same asL loc + quantile + bounds[0] * 0.01 + bounds[1] * 0.001
        torch.testing.assert_close(result, expected_results)


class TestICDF1D:
    """Testing rational: icdf_1d
    This uses an existing optimiser. We are not testing if the optimiser itselfs work, we are testing if we have
    configured it correctly, and it is suitable for our problem. We only test it on variant that we think are relative
    to our problem:


    Assumptions:

        - It is a valid CDF (Continous, monotonically increasing, bounded by [0,1]

    Cases relative to our problem:

        - Case 1: Optimisation in the extreme/tail region of distrbutions of standard distributions.
        - Case 2: Optimisation of a Mixture distribution (representing the marginal CDF)

            - Unit test: Function/line properties we could extect.

                1) Basic/smooth function [Covered in Case 1]
                2) Small discrete steps
                3) Location ands scale produced by a demo problem
                4) Incorrect bounds
                5) Dtype with insuffecient precision
                7)


            - Integration testing: We could do extensive testing by making combination of:

                - Distribution represetent the relationship between location and scale (e.g big loc, small scale)
                - make this function noise
                - See if the optimisation result reproduces the brute force

                NOTE: This is currently out of scope.


    Further tests requried and open questions:
        - Create test cases for:
            - scipy hitting x accuracy limit
            - optimisation terminating with insuffecient accuracy
            - optimisation clipping
        - what is the impact of using float32.
            - Do the number change significantly in test_icdf_1d_approximate_mixture_demo_problem_params. Compare this
              with
            the example in "demo/qoi_testing_extrapolation.py"
            - step size in q What happens if the optimise cant get to required precision because steps are too big
            - step size in x. What happens if the optimiser can get to the required accuracy because steps are too big
            - can there be a wide range of y values that return the same q. What result should we return in this case.
            - errors compound (e.g representation error +* max_acceptable_error), we don't consider how this impacts the
            total error
    """

    @pytest.mark.parametrize(
        "dist, q, maximum_timestep_error, bounds",
        [
            # Basic test - not realistic for our usecase
            (Gumbel(0, 1), 0.5, 1e-5, (-2, 5)),
            # Case 1: Optimisation in tail region. q - .5 at 20 years of 1 hour timesteps
            (
                Gumbel(0, 1),
                0.9999960436883025,
                acceptable_timestep_error(0.5, 20 * 365 * 24, atol=0.01),
                (-3, 30),
            ),
            (
                Normal(0, 1),
                0.9999960436883025,
                acceptable_timestep_error(0.5, 20 * 365 * 24, atol=0.01),
                (-30, 30),
            ),
        ],
    )
    def test_icdf_1d_basic_distribution(
        self, dist: Distribution, q: float, maximum_timestep_error: float, bounds: tuple[float, float]
    ):
        """Corresponds to "case 1" in the "testing rational".

        NOTE: Bounds are picked to be very wide without failing because they are outside the distribution numerical
        bounds.
        """
        actual_x = icdf_1d(dist, q, maximum_timestep_error, bounds)
        actual_q = dist.cdf(actual_x)

        assert actual_q.item() == pytest.approx(q, abs=maximum_timestep_error)

    @pytest.mark.integration
    def test_icdf_1d_approximate_mixture_step_function(self, *, visualise: bool = False):
        """Tests persormance on a challening Approximate Mixture distributiion.

        This is case 2.2 in test rational. This is a challening distribution that is effectly a step function.
        the desired quantil is effectively on the "corner" of one of the steps.
        """
        q_timestep = 0.9999960436883025  # equivalent to q_to_qtimestep(0.5, 20 * 365 * 24)
        maximum_timestep_error = acceptable_timestep_error(0.5, 20 * 365 * 24, atol=0.01)

        loc = torch.tensor([0, 1, 2, 3, 4, 10], dtype=torch.float64)
        scale = torch.ones_like(loc) * 0.001
        comp = Gumbel(loc, scale)
        mix = Categorical(torch.ones_like(loc, dtype=torch.float64))
        dist = ApproximateMixture(mix, comp)

        if visualise:
            x_domain = torch.linspace(0, 12, 100, dtype=torch.float64)
            y = dist.cdf(x_domain)
            _ = plt.plot(x_domain, y)
            _ = plt.xlabel("x")
            _ = plt.ylabel("CDF")
            plt.pause(5)

        # this brute force is accurate to 1e-10 q
        x_brute_force = torch.tensor(10.010648506485065, dtype=torch.float64)
        actual_x = icdf_1d(dist, q_timestep, maximum_timestep_error, bounds=(0, 20))
        # The actual value has some allowable error which means it does not have to be the brute force result exactly.
        # The below threshold is choosen arbitraily, the extent of the approximation is tested in the following assert/
        torch.testing.assert_close(actual_x, x_brute_force, rtol=0, atol=1e-6)

        # check the error
        q_error_actual = q_timestep - dist.cdf(actual_x)
        assert q_error_actual.abs().item() < maximum_timestep_error

    @pytest.mark.integration
    def test_icdf_1d_approximate_mixture_demo_problem_params(self):
        """Tests persormance on a challening Approximate Mixture distributiion.

        This is case 2.3 in test rational.

        NOTE: This is a good usecase for exploring the impacts of float32 as it appear to change the result
        significantly.
        """
        # define the required accuracy
        q_timestep = 0.9999960436883025  # equivalent to q_to_qtimestep(0.5, 20 * 365 * 24)
        maximum_timestep_error = acceptable_timestep_error(0.5, 20 * 365 * 24, atol=0.01)

        # create the distribution
        data_dir = Path(__file__).parent / "data" / "test_icdf"
        loc = torch.load(data_dir / "demo_loc.pt", weights_only=True).type(torch.float64)
        scale = torch.load(data_dir / "demo_scale.pt", weights_only=True).type(torch.float64)
        comp = Gumbel(loc, scale)
        mix = Categorical(torch.ones_like(loc, dtype=torch.float64))
        dist = ApproximateMixture(mix, comp)

        actual_x = icdf_1d(dist, q_timestep, maximum_timestep_error, bounds=(0, 100))

        x_brute_force = torch.tensor(61.025285285285285, dtype=torch.float64)
        # The actual value has some allowable error which means it does not have to be the brute force result exactly.
        # The below threshold is choosen arbitraily, the extent of the approximation is tested in the following assert/
        torch.testing.assert_close(actual_x, x_brute_force, rtol=0, atol=1e-5)

        # check the error
        q_error_actual = q_timestep - dist.cdf(actual_x)
        assert q_error_actual.abs().item() < maximum_timestep_error

    def test_icdf_1d_respects_maximal_error(self):
        # make a required error
        period_len = 20 * 365 * 24
        quantile = 0.5
        quantile_accuracy = 0.01
        q_timestep = q_to_qtimestep(quantile, period_len)
        maximum_timestep_error = acceptable_timestep_error(0.5, period_len, atol=quantile_accuracy)

        dist = Gumbel(torch.tensor([0], dtype=torch.float64), 1)

        actual_x = icdf_1d(dist, q_timestep, maximum_timestep_error, bounds=(0, 13))

        actual_q = dist.cdf(actual_x)

        assert actual_q**period_len == pytest.approx(quantile, abs=quantile_accuracy)

    def test_icdf_1d_incorret_bounds(self):
        """Optimiation should fail and return useful information when the bounds are incorrect."""
        dist = Gumbel(0, 1)
        q = 0.9999960436883025  # equaivalent to q_to_qtimestep(0.5, 20 * 365 * 24)
        maximum_timestep_error = acceptable_timestep_error(0.5, 20 * 365 * 24, atol=0.01)
        bounds = (20, 30)

        with pytest.raises(ValueError, match=re.escape("f(a) and f(b) must have different signs")):
            _ = icdf_1d(dist, q, maximum_timestep_error, bounds)

    def test_icdf_1d_incorrect_dtype(self):
        """Test the dtype being used is approapriate for the problem.

        The dtype used by torch can have a significant effect on the calculations. This is because the result is
        required to be accurate to many decimal place (the quantile is taken to the power many times). Specifically
        torch.float32 is typically not enough.

        General considertion for the optimisation:
            - Outputs don't have sufficient precision. (the dist is of the wrong dtype)
            - Input being optimised don't have suffecient precision

            These can cause it to fail due to not reaching the appropriate precision. They should be highlighted.

        Specific to scipy:
            - inputs are controlled by scipy and are always float. This is not a concern.
            - Outputs. Unclear what happen when unsuitable precision is used. TODO(sw)
        """

        loc = torch.tensor([0], dtype=torch.float32)
        scale = torch.ones_like(loc)
        dist = Gumbel(loc, scale)

        q = 0.9999960436883025  # equivalent to q_to_qtimestep(0.5, 20 * 365 * 24)

        with pytest.raises(TypeError, match="distribution provided does not have suitable resolution"):
            _ = icdf_1d(dist, q, 1e-10, (3, 13))
