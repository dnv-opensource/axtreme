# %%
import matplotlib.pyplot as plt
import pytest
import torch
from torch.distributions import Categorical, Distribution, Gumbel, LogNormal, MixtureSameFamily

from axtreme.distributions import ApproximateMixture
from axtreme.distributions.helpers import dist_cdf_resolution, mixture_cdf_resolution
from axtreme.distributions.mixture import icdf_value_bounds


# pyright: reportUnnecessaryTypeIgnoreComment=false
class TestApproximateMixture:
    """Testing overview: Approximate Mixture.

    Scope of testing:
        ApproximateMixture is a thin wrapper of MixtureSameFamily. We assume MixtureSameFamily test/gaurentees:

            - Produces valid CDFs and PDFs because:

                - The component distribution are gaurenteed to produce valid CDFs (from [0,1] with error outside of
                numerically supported range),
                - The mixture distribution (the weights) are gaurenteed to be in [0,1] and sum to 1.
                - The approach achieve suitable level of numerical accuracy.

        As such, the scope of testing is limited to the impact of the ApproximateMixture approximation/clipping.

    Test overview:
    - Appropriate lower/upper bound selction:
    - Running withing the bounds. Should have identical behaviour to MixtureSameFamily.
        - Single mixture.
        - batched ixture.
    - Does it perform correct clipping.
        - Large and small value expected.
        - The individual components are clipped to their specific values.
    - Weight:
        - too big and too small weights can be passed without causing issues


    Todo:
        The unit test rely on the above assumption. A developer could change the underlying implementation (so it is no
        longer ~identical to MixtueSameFamily) and break the method without any tests failing. Should implement the
        wider tests so this is not longer the case.

    """

    # TODO(sw 2024-12-14): Test more distributions
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("dist_class", [Gumbel, LogNormal])
    def test_show_lower_bound_x_issue(
        self,
        dtype: torch.dtype,
        dist_class: type[Distribution],
        *,
        visualise: bool = False,  # noqa: PT028
    ):
        """Mostly to demo behaviour that happens between q = [finfo.tiny, finfo.eps].

        Test fails if the lower bound is changed so developer will be made aware of the issues
        """
        dist = dist_class(torch.tensor([0], dtype=dtype), torch.tensor([1], dtype=dtype))  # type: ignore  # noqa: PGH003

        if visualise:
            # visualise the reverse relationship:
            finfo = torch.finfo(dtype)
            qs = torch.logspace(
                torch.log10(torch.tensor(finfo.tiny, dtype=torch.float32)),
                torch.log10(torch.tensor(0.00005, dtype=torch.float32)),
                100,
            )
            icdfs = dist.icdf(qs)

            _ = plt.axvline(finfo.eps, color="red", label="eps")
            _ = plt.axvline(finfo.tiny, color="blue", label="tiny")
            _ = plt.scatter(qs, icdfs, label="icdf(x) -> q")
            _ = plt.ylabel("x (icdf(q)-> x)")
            _ = plt.xlabel("q ")
            _ = plt.title("ICDF\n shows the relationship in reverse")
            _ = plt.xscale("log")
            _ = plt.legend()
            plt.pause(5)

            xs = icdfs
            q = dist.cdf(xs)
            _ = plt.scatter(xs, q)
            _ = plt.yscale("log")
            _ = plt.ylim(finfo.tiny, 5e-5)
            _ = plt.ylabel("q")
            _ = plt.xlabel("x ")
            _ = plt.title("CDF: cdf(x)->q")
            _ = plt.axhline(finfo.eps, color="red", label="eps")
            _ = plt.axhline(finfo.tiny, color="blue", label="tiny")
            plt.pause(5)

        mixture_dist = ApproximateMixture(Categorical(probs=torch.tensor([1], dtype=dtype)), dist)
        assert mixture_dist.quantile_bounds[0] == pytest.approx(torch.finfo(dtype).eps)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("dist_class", [Gumbel, LogNormal])
    def test_upper_bound_x(self, dtype: torch.dtype, dist_class: type[Distribution], *, visualise: bool = False):  # noqa: PT028
        """Mostly to demo behaviour that happens between q = [1 - finfo.eps, 1]."""
        dist = dist_class(torch.tensor([0], dtype=dtype), torch.tensor([1], dtype=dtype))  # type: ignore  # noqa: PGH003

        if visualise:
            finfo = torch.finfo(dtype)
            max_x = dist.icdf(torch.tensor(1 - finfo.eps, dtype=dtype))
            # NOTE: these bounds may need to be adjusted for different dists
            x = torch.linspace(max_x - 1, max_x + 0.5, 1000, dtype=dtype)
            cdfs = dist.cdf(x)

            _ = plt.axhline(1 - finfo.eps, color="red", label="eps")
            _ = plt.scatter(x, cdfs, label="cdf(x)-> q")
            _ = plt.ylabel("q (cdf(x)-> q)")
            _ = plt.xlabel("x")
            _ = plt.title("CDF\nNote TransfomedDist upper bound is 1 - finfo.eps, appears to work above this point")
            _ = plt.yscale("log")
            _ = plt.legend()
            plt.pause(5)

        mixture_dist = ApproximateMixture(Categorical(probs=torch.ones(1, dtype=dtype)), dist)
        assert mixture_dist.quantile_bounds[1] == pytest.approx(1 - torch.finfo(dtype).eps)

    @pytest.mark.parametrize(
        "component_distribution_shape, input_shape",
        [
            # unbatched mixture, unbatched input
            (torch.Size([3]), torch.Size([1])),
            # batched mixture, unbatched input
            (torch.Size([5, 3]), torch.Size([1])),
            # unbatched mixture, unbatched input
            (torch.Size([3]), torch.Size([7, 1])),
            # batched mixture, batched input
            (torch.Size([5, 3]), torch.Size([7, 5])),
            # larger_batch_mixture
            (torch.Size([10, 5, 3]), torch.Size([7, 10, 1])),
        ],
    )
    def test_cdf_shape_equivalent_to_mixture(self, component_distribution_shape: torch.Size, input_shape: torch.Size):
        """Check the input and output shape behaves the same way as MixtureSameFamily.

        TODO:
        - Test distribution with ``.event_shape!=torch.Size([])``
        """
        x = torch.rand(input_shape)

        loc = torch.zeros(component_distribution_shape)
        scale = torch.ones(component_distribution_shape)

        dist = Gumbel(loc, scale)
        mix = Categorical(torch.ones(component_distribution_shape))

        approx_mixture = ApproximateMixture(mix, dist)
        mixture = MixtureSameFamily(mix, dist)

        # Run through both of the methods.
        actual_value = approx_mixture.cdf(x)
        expected_value = mixture.cdf(x)

        assert actual_value.shape == expected_value.shape

    # Nothing interesting should happen with different shapes, just checking that here.
    @pytest.mark.parametrize(
        "component_distribution_shape",
        [
            # Unbatch mixture distribution, with only a single component
            torch.Size([1]),
            # Batched mixture distribution, with only a single component
            torch.Size([2, 1]),
            # batch, multiple components
            torch.Size([2, 3]),
        ],
    )
    @pytest.mark.parametrize(
        "q, dtype",
        [
            (0.5, torch.float64),  # point withing the range
            (torch.finfo(torch.float64).eps, torch.float64),  # smallest value before ApproximateMixture starts clipping
            # largest value before ApproximateMixture starts clipping
            (
                1 - torch.finfo(torch.float64).eps,
                torch.float64,
            ),
        ],
    )
    def test_cdf_equivalent_to_mixturesamefamily_within_range(
        self, component_distribution_shape: torch.Size, q: float, dtype: torch.dtype
    ):
        """ApproximateMixture and MixtureSameFamily should have identical behaviour within the component range.

        Within the range the component distribution supports, behaviour should be identical.
        NOTE: ApproximateMixture does NOT infact support the whole component range.

            - typical Component range: (finfo.tiny, 1 - finfo.eps)
            - ApproximateMixture range: (finfo.eps, 1 - finfo.eps)

        Below finfo.eps there are numerical issues with x != dist.cdf(dist.icdf(x)). Therefor ApproximateMixture clips
        this. See `docs/source/marginal_cdf_extrapolation.md` "Distribution lower bound issue" for details.

        This test checks:
            - ApproximateMixture and MixtureSameFamily have the same underling distributions
            - And the input is the in the range
            - The output should be identical
        """
        # Create the x value which represents that quantile. The gumbel here matches the one we use as dist.
        # This has shape (1,) which can be broadcast in the CDF step to any underlying distribution
        x = Gumbel(loc=torch.zeros(1, dtype=dtype), scale=1).icdf(q)

        loc = torch.zeros(component_distribution_shape, dtype=dtype)
        scale = torch.ones(component_distribution_shape, dtype=dtype)

        dist = Gumbel(loc, scale)
        mix = Categorical(torch.ones(component_distribution_shape, dtype=dtype))

        approx_mixture = ApproximateMixture(mix, dist)
        mixture = MixtureSameFamily(mix, dist)

        # Run through both of the methods.
        actual_value = approx_mixture.cdf(x)
        expected_value = mixture.cdf(x)

        assert torch.allclose(actual_value, expected_value)

    @pytest.mark.parametrize(
        "loc, scale, q, dtype, expect_violation",
        [
            # ## Standard case: loc and scale are close and this can be represented numerically
            (1, 1, 1 - torch.finfo(torch.float32).eps, torch.float32, False),
            (1, 1, 0.5, torch.float32, False),
            (1, 1, torch.finfo(torch.float32).eps, torch.float32, False),
            # Real failure: float32 can reliabilty represent the first 7 digits. For this problem would need 11+ digits
            (25_000, 1e-6, 1 - torch.finfo(torch.float32).eps, torch.float32, True),
            (25_000, 1e-6, 0.5, torch.float32, True),
            (25_000, 1e-6, torch.finfo(torch.float32).eps, torch.float32, True),
            # float64 is precise to 15 places.
            (25_000, 1e-6, 1 - torch.finfo(torch.float64).eps, torch.float64, False),
            # Float64: 10 digits for loc, 6 for scale -> leads to failure
            (25_000 * 10e5, 1e-6, 1 - torch.finfo(torch.float64).eps, torch.float64, True),
        ],
    )
    def test_check_cdf_numeric_precision(
        self,
        loc: float,
        scale: float,
        q: float,
        dtype: torch.dtype,
        expect_violation: bool,  # noqa: FBT001
    ):
        """Test can detect when don't have the precision to represent the bounds needed."""
        dist = Gumbel(loc=torch.tensor([loc], dtype=dtype), scale=torch.tensor([scale], dtype=dtype))
        results = ApproximateMixture._check_cdf_numeric_precision(dist, q)
        assert results["has_violation"] == expect_violation

    # @pytest.mark.parametrize(
    #     "x,expected_q",
    #     [
    #         # above bounds
    #         (100_000, 1 - torch.finfo(torch.float32).eps),
    #         # below bounds
    #         (-100_000, 0),
    #     ],
    # )
    # def test_cdf_value_outside_bounds(self, x: float, expected_q: float):
    #     dtype = torch.float32
    #     loc = torch.tensor([1], dtype=dtype)
    #     component_dist = Gumbel(loc=loc, scale=1)
    #     mixture_dist = Categorical(probs=torch.ones_like(loc))

    #     dist = ApproximateMixture(mixture_dist, component_dist)
    #     x_tensor = torch.tensor([x], dtype=dtype)
    #     cdf_value = dist.cdf(x_tensor)
    #     assert cdf_value == pytest.approx(expected_q)

    @pytest.mark.integration  # Relies on `dist_cdf_resolution` otherwise not really an integration test.
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_cdf_outside_of_component_range_single_component(self, dtype: torch.dtype):
        """Check the cdfs behaviour for x values outside of component range.

        This test uses a single underlying component distribution. This is not effected by numerical issues when taking
        weighted sum. As such finner grain tests of output can be performed. The test demonstrates the following:

            - The cdf does not fail for extremely larges of small inputs.
            - It produces the expected quantiles for these results (up to `dist_cdf_resolution` accuracy).
        """
        loc = torch.tensor([0.0], dtype=dtype)
        scale = torch.ones_like(loc, dtype=dtype)
        dist = Gumbel(loc, scale)
        mix = Categorical(torch.ones_like(loc, dtype=dtype))
        marginal = ApproximateMixture(mix, dist)

        # Find x values that have just entered the region consised outside of the component range
        finfo = torch.finfo(dist.mean.dtype)
        x_low = dist.icdf(torch.tensor(finfo.eps, dtype=dtype))
        x_high = dist.icdf(torch.tensor(1 - finfo.eps, dtype=dtype))

        max_cdf_value = marginal.cdf(x_high + x_high.abs() * 0.1)
        min_cdf_value = marginal.cdf(x_low - x_low.abs() * 0.1)

        assert min_cdf_value == pytest.approx(0, abs=dist_cdf_resolution(dist))
        assert max_cdf_value == pytest.approx(1 - finfo.eps, abs=dist_cdf_resolution(dist))

    @pytest.mark.integration  # Relies on `mixture_cdf_resolution` otherwise not really an integration test.
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_cdf_outside_of_component_range_batch_mixture(self, dtype: torch.dtype):
        """Check the cdfs behaviour for x values outside of component range.

        This uses a batch of component distributions. This is effected by numerical issues when taking
        weighted sum. The test demonstrates the following:

            - The cdf does not fail for extremely larges of small inputs  (across batches)
            - It produces the expected quantiles for these results (up to the accuracy specifed by
              `mixture_cdf_resolution`). NOTE: this is a less strict bound due to our uncertainty regarding impact of
              weighted sum numerical error.
        """
        n_posterior_samples = 2
        n_env_samples_per_period = 500

        # fmt: off
        loc = torch.zeros(n_posterior_samples, n_env_samples_per_period)*0.0
        scale = torch.ones_like(loc, dtype=dtype)
        mix = Categorical(torch.ones(n_env_samples_per_period, dtype=dtype))
        # fmt: on

        comp = Gumbel(loc=loc.type(dtype), scale=scale.type(dtype))
        marginal = ApproximateMixture(mix, comp)

        # Find x values that have just entered the region consised outside of the component range
        finfo = torch.finfo(comp.mean.dtype)
        x_low = comp.icdf(torch.tensor(finfo.eps, dtype=dtype)).min()
        x_high = comp.icdf(torch.tensor(1 - finfo.eps, dtype=dtype)).max()

        # make them compatable with the batch shape of the margina
        x_low = x_low.unsqueeze(-1)
        x_high = x_high.unsqueeze(-1)

        max_cdf_value = marginal.cdf(x_high)
        min_cdf_value = marginal.cdf(x_low)

        # The lower bounds can reach 0. Their can be numerical issues that come from the weighted sum.
        # We use
        torch.testing.assert_close(
            min_cdf_value,
            torch.zeros_like(min_cdf_value),
            atol=mixture_cdf_resolution(marginal),
            rtol=0,
        )
        torch.testing.assert_close(
            max_cdf_value,
            torch.ones_like(max_cdf_value) - finfo.eps,
            atol=mixture_cdf_resolution(marginal),
            rtol=0,
        )

    def test_cdf_input_is_uniquely_clipped_for_each_comp_dist(self):
        """The input need to be clipped for the specific range each underling component supports.

        Here we check the input is clipped for one distribution and not the other. This does not cover all cases.
        """

        dtype = torch.float64
        finfo = torch.finfo(dtype)

        loc = torch.tensor([0.0, 100], dtype=dtype)
        scale = torch.ones_like(loc, dtype=dtype)
        dist = Gumbel(loc, scale)

        # find the input x to use. It should produce:
        # for the first component: cdf(x) = 1 - finfo.eps
        # for the second component: cdf(x) = .5
        x = dist.icdf(0.5)[1]

        mix = Categorical(torch.ones_like(loc))
        marginal = ApproximateMixture(mix, dist)

        expected_result_comp1 = 1 - finfo.eps
        expected_result_comp2 = 0.5
        expected_result = (expected_result_comp1 + expected_result_comp2) / 2
        actual_result = marginal.cdf(x).item()

        assert actual_result == pytest.approx(expected_result, abs=finfo.resolution)

    @pytest.mark.parametrize(
        "weights",
        [
            # total probability less than 1
            (torch.tensor([0.5, 0.3], dtype=torch.float64)),
            # total probability greater than 1
            (torch.tensor([0.5, 0.8], dtype=torch.float64)),
        ],
    )
    def test_weights_not_summing_to_one(self, weights: torch.Tensor):
        """The weights provided may be inapproapriate, check they are handled.

        Weight provided (e.g through importance sampleing) might not sum to one. Check that this distribution handles
        this and call still produce a valid distribution.

        Args:
            weights: 1d vector of weights to use in the mixture
        """
        dtype = weights.dtype

        loc = torch.zeros_like(weights)
        scale = torch.ones_like(weights, dtype=dtype)

        dist = Gumbel(loc, scale)
        mix = Categorical(probs=weights)
        marginal = ApproximateMixture(mix, dist)

        # check that appropraite CDF results are achieved
        very_large_value = torch.tensor([100_000], dtype=dtype)  # we know this is outside the bounds
        very_small_value = torch.tensor([-100_000], dtype=dtype)  # we know this is outside the bounds

        finfo = torch.finfo(dtype)
        assert marginal.cdf(very_large_value) == pytest.approx(1 - finfo.eps, abs=finfo.resolution)
        assert marginal.cdf(very_small_value) == pytest.approx(finfo.eps, abs=finfo.resolution)

    def test_cdf_input_should_have_float64_dtype(self):
        """Demonstrate how float32 input can produce different result (due to numerical issues).

        NOTE: The size of the numerical issues here are probably not large enough to cause concern
        (e.g  acceptable_timestep_error(0.5, 20 * 365 * 24, atol=0.01) = 1.13e-7), BUT we are not confident we have
        identfied the scenario causes the largest possible error.
        """
        dtype = torch.float64

        loc = torch.tensor([0.0], dtype=dtype)
        scale = torch.ones_like(loc, dtype=dtype)
        dist = Gumbel(loc, scale)
        mix = Categorical(torch.ones_like(loc))
        marginal = ApproximateMixture(mix, dist, check_dtype=False)

        x_32 = torch.tensor([0.1], dtype=torch.float32)
        x_64 = torch.tensor([0.1], dtype=torch.float64)

        q_32 = marginal.cdf(x_32)
        q_64 = marginal.cdf(x_64)

        with pytest.raises(AssertionError):
            # difference is 5.45e-10
            torch.testing.assert_close(q_32, q_64, atol=1e-10, rtol=0)

    def test_calculate_x_bounds_returns_correct_values(self):
        """calculate_x_bounds should return dist.icdf(q_lowerbound) and dist.icdf(q_upperbound)."""
        dtype = torch.float64
        finfo = torch.finfo(dtype)
        q_lower = finfo.eps
        q_upper = 1 - finfo.eps

        loc = torch.tensor([0.0], dtype=dtype)
        scale = torch.ones_like(loc)
        dist = Gumbel(loc, scale)

        x_lower, x_upper = ApproximateMixture.calculate_x_bounds(dist, q_upperbound=q_upper, q_lowerbound=q_lower)

        torch.testing.assert_close(x_lower, dist.icdf(torch.tensor(q_lower, dtype=dtype)))
        torch.testing.assert_close(x_upper, dist.icdf(torch.tensor(q_upper, dtype=dtype)))

    def test_calculate_x_bounds_warns_on_precision_violation(self):
        """calculate_x_bounds should emit a RuntimeWarning when the dtype lacks precision to represent the bounds.

        Uses loc=25_000, scale=1e-6 with float32 — a case proven to trigger violations in
        _check_cdf_numeric_precision (both upper and lower bounds).
        """
        dtype = torch.float32
        finfo = torch.finfo(dtype)
        q_lower = finfo.eps
        q_upper = 1 - finfo.eps

        loc = torch.tensor([25_000.0], dtype=dtype)
        scale = torch.tensor([1e-6], dtype=dtype)
        dist = Gumbel(loc, scale)

        with pytest.warns(RuntimeWarning, match="Insufficient precision"):
            ApproximateMixture.calculate_x_bounds(dist, q_upperbound=q_upper, q_lowerbound=q_lower)


@pytest.mark.parametrize(
    "q, bounds1, bounds2",
    [
        # a simple q applied to each mixture model
        (
            torch.tensor(0.5),
            # This are known upfront because of the loc and scale specified in the test
            (Gumbel(0, 1).icdf(0.5), Gumbel(1, 1).icdf(0.5)),  # expected bound for 1s mixture model
            (Gumbel(1, 1).icdf(0.5), Gumbel(2, 1).icdf(0.5)),  # expected bound for 2nd mixture model
        ),
        # Mixture model specific quantiles
        (
            torch.tensor([0.5, 0.9]),
            # This are known upfront because of the loc and scale specified in the test
            (Gumbel(0, 1).icdf(0.5), Gumbel(1, 1).icdf(0.5)),  # expected bound for 1s mixture model
            (Gumbel(1, 1).icdf(0.9), Gumbel(2, 1).icdf(0.9)),  # expected bound for 2nd mixture model
        ),
    ],
)
def test_icdf_value_bounds(q: torch.Tensor, bounds1: tuple[float, float], bounds2: tuple[float, float]):
    """Tests that specific bounds are generated for each of the underling mixture distributions.

    Check that the quantile is appropriately broacdcast.

    NOTE: Bounds are specified verbosely in areguments to make it clearer where they come from. Could simplfy this to
    just a tensor of bounds.
    """
    # Set to the require components
    # fmt: off
    loc = torch.tensor([[0.0, 1.,], [1.0, 2.,]], dtype=torch.float64)
    # fmt: on
    shape = torch.ones_like(loc)
    comp = Gumbel(loc=loc, scale=shape)
    mix = Categorical(torch.ones_like(loc, dtype=torch.float64))
    dist = MixtureSameFamily(mix, comp)

    # Run the test
    bounds = icdf_value_bounds(dist, q)

    # The bounds are clearly distinct, so we don't need to use high precision testing
    actual_bouds1 = bounds[:, 0]
    torch.testing.assert_close(actual_bouds1, torch.tensor(bounds1), atol=1e-5, rtol=0)
    actual_bouds2 = bounds[:, 1]
    torch.testing.assert_close(actual_bouds2, torch.tensor(bounds2), atol=1e-5, rtol=0)


@pytest.mark.parametrize(
    "loc",
    [
        # Negative loc: check not using multiplication/division which will invert bounds.
        -10.0,
        # Near-zero loc: offset must not collapse to 0
        0.0,
        # Positive loc: sanity check that the fix doesn't break the normal case
        10.0,
        # Large loc: check the offset is not
        1e12,
    ],
)
def test_icdf_value_bounds_identical_components_lower_lt_upper(loc: float):
    """Handles edge case of picking bounds when components all give identical icdf value (e.g. components are identical)

    Check that in all cases lower_bounc < upper_bound.
    """
    q = torch.tensor(0.5)
    identical_loc = torch.tensor([loc, loc], dtype=torch.float64)
    scale = torch.ones_like(identical_loc)
    comp = Gumbel(loc=identical_loc, scale=scale)
    mix = Categorical(torch.ones_like(identical_loc, dtype=torch.float64))
    dist = MixtureSameFamily(mix, comp)

    bounds = icdf_value_bounds(dist, q)
    lower, upper = bounds[0], bounds[1]

    assert lower < upper, f"Expected lower < upper for loc={loc}, got lower={lower}, upper={upper}"
