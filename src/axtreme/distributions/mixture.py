"""Mixture model variants."""

import warnings

import torch
from torch.distributions import Categorical, Distribution, MixtureSameFamily

from axtreme.distributions.utils import dist_dtype


class ApproximateMixture(MixtureSameFamily):
    """Mixture distributions where extreme calculations are approximated.

    Some distribution only support a limited range of quantiles (e.g :math:`[.01, 99]`) due to numerical issues (see
    "Details"). When calculation such as :math:`q=cdf(x)` or :math:`x=icdf(q)` fail outside this range the function then
    error. ``ApproximateMixture`` allows all x values ,and approximates the results for x values outside the supported
    range (see "Details" for approximation method).

    Details:

        Distribution Quantiles Bounds:

            Some distributions (e.g `TransformedDistribution`) have bounds on the quantiles they can support.
            Calculations such as :math:`q=cdf(x)` or :math:`x=icdf(q)` will fail when q is outside of these bounds.
            Bounds exist because for sum distributions :math:`icdf(1)=inf` or :math:`icdf(0)=-inf`. Eventually there
            comes a point for q value close to 0 or 1 where they lack the numerical precision to capture very small
            changes in q, and the very large values of x.

        Approximation principles:

            It is assumed (:math:`1-cdf(x)`) represent the chance of failure, and we want to avoid underestimating this.
            For example, x could represent the strength a structure is designed to withstand, and
            (:math:`1-cdf(x)`) represents the chance of experiencing a force that will break the structure (e.g the
            risk). It is better to overestimate the risk (resulting in a conservative design), rather than underestimate
            risk. In other words:

            TLDR:

                - :math:`cdf_est(x) < cdf_true(x)`: produces conservative design (okay)
                - :math:`cdf_est(x) > cdf_true(x)`: BAD

            Worked example:

                - :math:`cdf_est(x) = .5` and :math:`cdf_true(x) = .4`
                - If structure is designed to be :math:`x` strong, then estimated number of failure is .5, true number
                  of failures is .6.
                - Have underestimated the risk and designed a structure more likely to fail than we expect.

        Approximate results.

            ApproximateMixture provides exact results within the quantile bounds ``[finfo.eps, 1 - finfo.eps]`` (where
            ``finfo = torch.finfo(component_distribution.dtype)``). NOTE: while ``TransformedDistribution`` says it
            supports ``[finfo.tiny, 1 - finfo.eps]``, behaviour is unreliable below ``finfo.eps``
             (see `tests/distributions/test_mixture.py:test_lower_bound_x' ). When outside these bounds:
             - x > icdf(1 - finfo.eps): we return :math:`cdf(x) = 1 - finfo.eps` (conservative estimate)
             - x < icdf(finfo.eps): we return :math:`cdf(x) = 0` (conservative estimate)

            See 'Impact of Approximation' for details of why this is conservative.

        Impact of Approximation:

            As a result of the approximation, the cdf method can underestimate the analytical cdf value by up to
            ``finfo.eps``. This values come from the following calculation, Similar behaviour occurs at the lower bound.

                - Consider one of the underlying distributions in the marginal: once x hits the upper bound

                    - it will give q = 1.0-finfo.eps
                    - in reality it could be as high as 1.0
                    - so the q error is finfo.eps(at worst)

                - The marginal cdf is calculated from the underlying distributions:

                    - e.g :math:`q_marginal = w_1 * q_1 + ... + w_n * q_n`
                    - :math:`sum(w_i) = 1`

                - In the worst case:

                    - Underling distribution: q give is finfo.eps too small (at worst)
                    - weight = 1
    """

    def __init__(
        self,
        mixture_distribution: Categorical,
        component_distribution: Distribution,
        *,
        validate_args: bool | None = None,
        check_dtype: bool = True,
    ) -> None:
        """Initialise the ApproximateMixture.

        Args:
            mixture_distribution: ``torch.distributions.Categorical``-like instance. Manages the probability of
              selecting component. The number of categories must match the rightmost batch dimension of the
              ``component_distribution``. Must have either scalar ``batch_shape`` or ``batch_shape`` matching
              ``component_distribution.batch_shape[:-1]``
            component_distribution: ``torch.distributions.Distribution``-likeinstance. Right-most batch dimension
              indexes component.
            validate_args: Runs the default validation defined by torch
            check_dtype: Check datatype of distributions match and inpput are at least as precise as the distribution.
        """
        super().__init__(mixture_distribution, component_distribution, validate_args=validate_args)
        self.check_dtype = check_dtype

        component_dtype = dist_dtype(component_distribution)
        mixture_dtype = dist_dtype(mixture_distribution)
        if check_dtype and (component_dtype != mixture_dtype):
            msg = (
                "Component and mixture distributions should be of the same dtype, otherwise results could have the"
                f" resolution of the lowest dtype. Got dtypes {component_dtype} and {mixture_dtype} respectively"
            )
            raise TypeError(msg)
        self.dtype = component_dtype

        finfo = torch.finfo(component_dtype)

        # NOTE: Even though internally TranformedDistribution sets its lowerbound with finfo.tiny, it appears to have
        # issues below finfo.eps. See `docs\source\marginal_cdf_extrapolation.md` "Distribution lower bound issue"
        # TODO(sw 2026-04-08): The referenced doc is missing, find it or update the link
        self.quantile_bounds = torch.tensor([finfo.eps, 1 - finfo.eps], dtype=component_dtype)
        # the q value to be returned when the cdf is run with an x that is outside the supported q range (lower, upper)
        # See "Approximation principles" for why these values are selected.
        self.cdf_out_of_bounds_q = torch.tensor([0.0, 1 - finfo.eps], dtype=component_dtype)
        # NOTE: bounds are stored here so we don't need to be recomputed every `.pdf()` or `.cdf()` call.
        # Can put in the call if we want to save memory.
        self.x_bound_lower, self.x_bound_upper = ApproximateMixture.calculate_x_bounds(
            dist=component_distribution,
            q_upperbound=self.quantile_bounds[1].item(),
            q_lowerbound=self.quantile_bounds[0].item(),
            weights=mixture_distribution.probs,
        )

    @staticmethod
    def _check_cdf_numeric_precision(
        dist: Distribution,
        q: float,
        weights: torch.Tensor | None = None,
    ) -> dict[str, bool | float | None | torch.Tensor]:
        """Helper to check if the dtype can represent the result ICDF(q)=x value with enough precision (so CDF(x)=q).

        We store the x bounds associated with quantile bounds. We need to ensure we have enough numerical precision to
        store these bounds accurately. This can often fail if location and scale have very different values.

        Example:
            ```python
            finfo32 = torch.finfo(torch.float32)
            dist = Gumbel(loc=torch.tensor(25000, dtype=torch.float32), scale=torch.tensor(1e-6, dtype=torch.float32))
            top_eps_quantile = dist.icdf(torch.tensor(1 - finfo32.eps))
            print(top_eps_quantile.item())  # 25000.0
            # Problem: top_eps_quantile can't be represented with enough accuracy
            print(dist.cdf(top_eps_quantile).item())  # -> .3678 not 1-finfo.eps
            ```

        Args:
            dist: The distribution with batch_shape (*b)
            q: The quantile to check the bound for.
            weights: (shape (*b,)) The weights of each distribution in the mixture (if applicable).

        Returns:
            dict with keys:
                - "has_violation": bool, if any of the batch has precision issue
                - "percent_violating": float, percentage of the batch that has precision issue
                - "mean_violation_q": float, the mean q value calculated for the batch that has precision issue
                   (should be close to q)
                - "total_weight_of_violations": tensor or None, the total weight of the items in the batch with issues.
        """
        dtype = dist_dtype(dist)
        q_expected = torch.tensor(q, dtype=dtype)
        # Appears to always produce x value inside the supported q range (even with numeric issues). Assume this is the
        # case, and if not we will get an error and can update.
        x_value = dist.icdf(q_expected)
        q_actual = dist.cdf(x_value)

        # As value is between (0,1), should be correct with eps/2, but allow some small buffer
        bound_violation_mask = ~torch.isclose(q_actual, q_expected, atol=torch.finfo(dtype).eps, rtol=0.0)

        percent_violating = bound_violation_mask.to(torch.float).mean().item()
        mean_violation_q = q_actual[bound_violation_mask].mean().item()

        total_weight_of_violations = None
        if weights is not None:
            weights_expanded: torch.Tensor = weights.clone().expand_as(q_actual)  # expand if required
            total_weight_of_violations = weights_expanded[bound_violation_mask].sum(dim=-1)

        return {
            "has_violation": bound_violation_mask.any().item(),
            "percent_violating": percent_violating,
            "mean_violation_q": mean_violation_q,
            "total_weight_of_violations": total_weight_of_violations,
        }

    @staticmethod
    def calculate_x_bounds(
        dist: Distribution,
        q_upperbound: float,
        q_lowerbound: float,
        weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Check the dtype has suitable numeric precision to represent the bound with the required precision.

        We store the x bounds associated with quantile bounds. We need to ensure we have enough numerical precision to
        store these bounds accurately. This can often fail if location and scale have very different values.

        Example:
            ```python
            finfo32 = torch.finfo(torch.float32)
            dist = Gumbel(loc=torch.tensor(25000, dtype=torch.float32), scale=torch.tensor(1e-6, dtype=torch.float32))
            top_eps_quantile = dist.icdf(torch.tensor(1 - finfo32.eps))
            print(top_eps_quantile.item())  # 25000.0
            # Problem: top_eps_quantile can't be represented with enough accuracy
            print(dist.cdf(top_eps_quantile).item())  # -> .3678 not 1-finfo.eps
            ```
        Note:
        Potential fixes for this issues include:
        - Use a higher precision dtype
        - Reduce the magnitude of difference between location and scale (failure is not significantly effected by q)

        Args:
            dist: The distribution with batch_shape (*b)
            q_upperbound: The upper quantile bound to check (e.g 1-finfo.eps)
            q_lowerbound: The lower quantile bound to check (e.g finfo.eps)
            weights: (shape (*b,)) The weights of each distribution in the mixture (if applicable).

        Returns:
        Upper and lower x bounds corresponding to the given quantiles (shape (*b,)).

        Dev notes:
        - This issue is partly remedied in `cdf` because values that are outside x_bounds have results calculated
          manually (not useing the cdf function). For example for the upper bound, the exact location of the x_bounds is
          not correct, but values above this bound will return cdf(x)=1-eps (rather than e.g. .3678)
        """
        upper_bound_violation_statistics = ApproximateMixture._check_cdf_numeric_precision(dist, q_upperbound, weights)
        lower_bound_violation_statistics = ApproximateMixture._check_cdf_numeric_precision(dist, q_lowerbound, weights)

        for bound_name, stats in [
            ("upper", upper_bound_violation_statistics),
            ("lower", lower_bound_violation_statistics),
        ]:
            if stats["has_violation"]:
                q_bound = q_upperbound if bound_name == "upper" else q_lowerbound
                cdf_direction = "smaller" if bound_name == "upper" else "LARGER"
                estimate_type = "more conservative" if bound_name == "upper" else "NON-conservative"

                msg = (
                    f"Insufficient precision to represent {bound_name} bound ({estimate_type} result).\n"
                    f"When insufficient precision is used, the x_bound value associated with q_{bound_name}bound "
                    f"cannot be stored with enough precision, as a result cdf(x_bound)!=q_{bound_name}bound). "
                    f"cdf(x_bound) will produce a {cdf_direction} value than q_{bound_name}bound, which will "
                    f"lead to a {estimate_type} cdf estimate. See ApproximateMixture "
                    "`Approximation principles` for more details.\n"
                    "Possible fixes:\n"
                    " - Use a higher precision dtype\n"
                    " - Reduce the magnitude of difference between location and scale\n"
                    "Violation statistics:\n"
                    f"percent of batch violating: {stats['percent_violating']:.2f}%\n"
                    f"average result of cdf(x_bound) for violating batch (should be {q_bound}): "
                    f"{stats['mean_violation_q']:.2e}\n"
                )

                if weights is not None:
                    msg += f"total weight of violating batch: {stats['total_weight_of_violations']:.2e}\n"

                warnings.warn(msg, category=RuntimeWarning, stacklevel=1)

        dtype = dist_dtype(dist)
        x_bound_lower = dist.icdf(torch.tensor(q_lowerbound, dtype=dtype))
        x_bound_upper = dist.icdf(torch.tensor(q_upperbound, dtype=dtype))
        return x_bound_lower, x_bound_upper

    def cdf(self, x: torch.Tensor) -> torch.Tensor:
        """Return the CDF.

        Identical to MixtureSameFamily implementation except for clamping.

        Args:
            x: Values to calculate the CDF for. Must be broadcastable with the ``ApproximateMixture.batch_shape``. E.g

                - ``self.component_distribution.batch_shape = (2,5)`` (last dimension is the components that are
                  combined to make a single Mixture distribution)
                - ``self.batch_shape = (2,)``
                - ``x.shape = (10)`` This will fail as it is not broadcastable
                - ``x.shape = (10,1)`` This will pass as it is broadcastable

        Returns:
            Tensor of shape (x.shape[:-1], self.batch_shape)
        """
        if self.check_dtype and torch.finfo(x.dtype).resolution > torch.finfo(self.dtype).resolution:
            msg = (
                f"Input type {x.dtype} is less precise than dtype of the distribution {self.dtype}."
                " This can lead to loss of precision."
            )
            raise TypeError(msg)
        x = x.clone()
        # Expected x already broadcasts to `self.batch_dimension` (same as MixtureSameFamily)
        # The final expanded x shape need to be (*b, component_dist.batch_shape, component_dist.event_shape)
        # where masking will be done on out of bounds values across component_dist.batch_shape
        x_batch_shape = x.shape[: -len(self.batch_shape)] if self.batch_shape != torch.Size([]) else x.shape
        # resulting shape (*b, component_dist.batch_shape[:-1],1, component_dist.event_shape)
        x = self._pad(x)
        # resulting shape
        x = x.expand(x_batch_shape + self.component_distribution.batch_shape + self.component_distribution.event_shape)

        # find where input is outside the allowed range
        # bounds have shape (component_dist.batch_shape, component_dist.event_shape)
        too_large_mask = x > self.x_bound_upper
        too_small_mask = x < self.x_bound_lower

        # replace out of bounds values with a placeholder value
        dummy_value = self.component_distribution.mean  # shape (component_dist.batch_shape, component_dist.event_shape)
        # Convert to x.dtype. Once .cdf() runs the output will conform to MixtureSameFamily (biggest dtype used)
        dummy_value = dummy_value.to(x.dtype)
        x = x.clone()
        x[too_large_mask | too_small_mask] = dummy_value.expand_as(x)[too_large_mask | too_small_mask]

        cdf_x = self.component_distribution.cdf(x)

        # Manually replaces the result of the out of bound CDF calc (follow MixtureSafeFamily for dtype convention)
        cdf_x[too_large_mask] = self.cdf_out_of_bounds_q[1].expand_as(x)[too_large_mask].to(cdf_x.dtype)
        cdf_x[too_small_mask] = self.cdf_out_of_bounds_q[0].expand_as(x)[too_small_mask].to(cdf_x.dtype)

        mix_prob = self.mixture_distribution.probs
        # This is the approach found in torch. Can introduce small numerical errors, but we assume these are neglible.
        return torch.sum(cdf_x * mix_prob, dim=-1)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate the log prob (e.g log(pdf)).

        Args:
            x: Values to calcuate the log prob for. Must be broadcastable with the ``ApproximateMixture.batch_shape``.
              E.g

                - ``self.component_distribution.batch_shape = (2,5)`` (last dimension is the components that are
                  combined to make a single Mixture distribution)
                - ``self.batch_shape = (2,)``
                - ``x.shape = (10)`` This will fail as it is not broadcastable
                - ``x.shape = (10,1)`` This will pass as it is broadcastable

        Returns:
            Tensor of shape (x.shape[:-1], self.batch_shape)
        """
        # Dev notes:
        # NOTE: MixtureSameFamily.log_prob makes use of `_validate_sample(x)`
        # There are challenges with the shape here because:
        # - Background:
        #     - Mixture has shape (*b,c) where:
        #         - *b: is batch dim, each of them representing a different mixture dist
        #         - c: is the component dists that get summed together to make a mixture
        #     - X has shape (*s,*b) where:
        #         - *s: batches of data to evaluate with each mixture dist
        #         - *b: matches the batch (number of models) of the mixture (or can be broadcast)
        #     e.g: to feed 10 input to 2 different mixtures the shapes would be
        #         - Mixture: (b  = 2, c = n)
        #         - x = (s = 10, b = 1)  -- the batch dimesnions can boudcast
        # - Our context:
        #     - Each x item need to be cliped to the range of the component distribution `c`
        #     - x_clipped = (*s,*b,c)
        #     - This shape is not compatible with _validate_sample(clipped_x)
        #         - this expects shape (*b)
        #             - First it first does check to make sure we get a specific shape (irrelevant to us)
        #             - then it broadcast this to (*s,*b,c) and checks if within supported range

        # - required decision:
        #     - Decide if we need to use _validate_sample, and if so, which parts.
        raise NotImplementedError


def icdf_value_bounds(dist: MixtureSameFamily, q: torch.Tensor) -> torch.Tensor:
    r"""Returns bounds in which the value for quantile q is gaurenteed to be found.

    Args:
        dist: ``(*batch_shape,)`` mixture distribution producing events of event_shape samples.
        q: quantile to find the inverse cdf of. Must be boardcastable up to (``*batch_shape``,). Must not have more
           dimensions than ``*batch_shape`` (only 1 q can be passed to each of the distributions in the batch.)

    Returns:
        tensor of shape (2,*batch_shape), there the first index represents the lower
        bounds, and the second the upper bounds.

    Details:
    Mixture distribution calculate the CDF as follows:

        :math:`q = w_i * CDF_1(y) + w_2 * CDF_2(y) + ... + w_n * CDF_n(y)`
        Which can be written as: :math:`q = w_i * q_1 + w_2 * q_2 + ... + w_n * q_n`

        where :math:`0 <= w_i <= 1` and :math:`\\sum{w_i} = 1`

    An effective way to bound the x values the can produce y is:

        - take the ``icdf(q)`` for each distribution. Now have X_n values.
        - ``lower_bound = min(X_n)``: at this point the first component distribution has become big enough to produce q.
          As the weights are between [0,1] no point prior would be able to procude q as no q_i was large enough.s
        - ``upper_bound = max(X_n)``: at this point the last component distribution has become big enough to produce q.
          As the weights sum to one, q must be produced by this point.
    """
    # expand can not be done to smaller shapes, so this safely handled the case when q.dim() > len(dist.batch_shape)
    q_expanded = q.expand(dist.batch_shape)

    q_expanded = q_expanded.unsqueeze(-1)  # add a dimension for the underlying component distributions.
    values = dist.component_distribution.icdf(q_expanded)
    lower_bound = values.min(dim=-1).values
    upper_bound = values.max(dim=-1).values

    # Handle edge cases where the lower and upper bound are the same.
    # NOTE: This only sets the starting bounds for optimisation, so its not an issue if the bounds are a little large.
    identical = torch.isclose(lower_bound, upper_bound)
    # If working with very small number eps with will comparatively large - but should still find correct result.
    finfo_eps = torch.finfo(lower_bound.dtype).eps
    # large numbers: a difference of eps will be truncated by numeric precision, so find a relative value
    # Small number: Could use a value smaller than eps, but don't bother
    offset = (lower_bound.abs() * finfo_eps * 10).clamp(min=finfo_eps)
    lower_bound = torch.where(identical, lower_bound - offset, lower_bound)
    upper_bound = torch.where(identical, upper_bound + offset, upper_bound)

    return torch.stack([lower_bound, upper_bound])
