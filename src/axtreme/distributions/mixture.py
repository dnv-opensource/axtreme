"""Mixture model variants."""

import torch
from torch.distributions import Categorical, Distribution, MixtureSameFamily

from axtreme.distributions.utils import dist_dtype


class ApproximateMixture(MixtureSameFamily):
    """Mixture distributions where extreme caclulations are approximated.

    Some distribution only support a limited range of quantiles (e.g [.01, 99]) due to numerical issues (see "Details").
    When calculation such as `q=cdf(x)` or `x=icdf(q)`fail outside this range. ApproximateMixture allows all x values,
    and approximates the results for x values outside the supported range (see "Details" for approximation method).

    Details:

        Distribution Quantiles Bounds:

            Some distributions (e.g `TransformedDistribution`) have bounds on the quantiles they can support.
            Calculations such as `q=cdf(x)` or `x=icdf(q)` will fail when q is outside of these bounds. Bounds exist
            because for sum distributions `icdf(1)=inf` or `icdf(0)=-inf`. Eventually there comes a point for q value
            close to 0 or 1 where they lack the numerical precision to capture very small changes in q, and the very
            large values of x.

        Approximation principles:

            It is assumed we want to conservatively estimate (1-cdf(x)), the chance the a value will exceed some
            level `x`. For example, x could represent the strength a structure is designed to withstand, and (1-cdf(x))
            represents the chance of experiencing a force that will break the structure (e.g the risk). It is better to
            overestimate the risk (conservative), rather than underestimate it. In other words, if the `cdf_est(x)` over
            estimates the `cdf_true(x)`, then the true risk of exceeding x is. e.g:

            TLDR:

                - `cdf_est(x)` < `cdf_true(x)`: produces conservative design (okay)
                - `cdf_est(x)` > `cdf_true(x)`: BAD

    Example:
                - `cdf_est(x) = .5` and `cdf_true(x) = .4`
                - If structure is designed to be `x` strong, then estimated number of failure is .5, true number of
                  failures is .6.
                - Have underestimated the risk and designed a structure more likely to fail than we expect.

        Approximate results.

            ApproximateMixture provides exact results within the quantile bounds `[finfo.eps, 1 - finfo.eps]` (where
            `finfo = torch.finfo(component_distribution.dtype)`) For details regarding why this range is selected see
            `_lower_bound_x` and `_upper_bound_x`. Values smaller than `finfo.eps` are approximated
            according to `_lower_bound_x`. Values larger than `1 - finfo.eps` are approximated according to
            `_upper_bound_x`. Note the range `[finfo.eps, 1 - finfo.eps]` is still used even if the component
            distribution supports a greater range.

        Impact of Approximation:

            As a result of the approximation, the cdf method can underestimate the analytical cdf value by up to
            `finfo.eps`. This values come from the following calculation, Similar behaviour occurs at the lower bound.

                - Consider one of the underlying distributions in the marginal: once x hits the upper bound

                    - it will give q = 1.0-finfo.eps
                    - in reality it could be as high as 1.0
                    - so the q give is finfo.eps too small (at worst)

                - The marginal cdf is calcualted from the underlying distributions:

                    - e.g `q_marginal = w_1 * q_1 + ... + w_n * q_n`
                    - `sum(w_i) = 1`

                - In the worst case:

                    - Underling distribution: q give is finfo.eps too small (at worst)
                    - weight = 1

    Notes:
        It is worth ensuring that the ApproximateMixture distribution has suitable numeric precision for its intended
        use. See `axtreme.distributions.utils.mixture_dtype` for more details.
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
            mixture_distribution: `torch.distributions.Categorical`-like instance. Manages the probability of selecting
                component. The number of categories must match the rightmost batch dimension of the
                `component_distribution`. Must have either scalar `batch_shape` or `batch_shape` matching
                `component_distribution.batch_shape[:-1]`
            component_distribution: `torch.distributions.Distribution`-likeinstance. Right-most batch dimension indexes
                component.
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
        self.quantile_bounds = torch.tensor([finfo.eps, 1 - finfo.eps], dtype=component_dtype)
        # NOTE: bounds are stored here so we don't need to be recomputed every `.pdf()` or `.cdf()` call.
        # Can put in the call if we want to save memory.
        self.x_bound_lower = ApproximateMixture._lower_bound_x(component_distribution)
        self.x_bound_upper = ApproximateMixture._upper_bound_x(component_distribution)

    @staticmethod
    def _lower_bound_x(dist: Distribution) -> torch.Tensor:
        """Returns the x lowerbound for each of the component distributions.

        If `cdf(x)` recieves values that are too small, it throws an error. We instead find a suitable lower bound and
        later clip the x values to this range. The lower bound should:

            - Not throw an error when it is used in `cdf(lower_bound).`
            - As per "Approximation principles" it should not result in overestimate of the analytical cdf.

        Args:
            dist: The component distribution.

        Return:
            Tensor of shape (dist.batch_shape).

        Details:
            TransformedDistribution are officially bounded between `(finfo.tiny, 1 - finfo.eps)`. The CDF does not match
            the mathematical CDF across this whole region (due to numeric errors) - Specifically the region
            (finfo.tiny, finfo.eps).

            in the region (finfo.tiny, finfo.eps):

                - cdf(x): will output 0 for most of the region
                - cdf(x): x corresponding q = finfo.tiny will produces error about being outside of the allowed range,
                  due to numerical issues. `x` in the region will produce the same error.

            Solution: Pick an x value corresponsing to the middle q range (finfo.tiny, finfo.eps). This should produce
            cdf(x)=0, which underestimates the true cdf value by up to `finfo.eps`. As per "Approximation principles"
            this is acceptable.
        """
        dtype = dist_dtype(dist)
        finfo = torch.finfo(dtype)
        tiny = torch.tensor(finfo.tiny, dtype=dtype)
        eps = torch.tensor(finfo.eps, dtype=dtype)

        # See unit test visualisation for whne a q between tiny and eps can't be used directly
        return (dist.icdf(tiny) + dist.icdf(eps)) / 2

    @staticmethod
    def _upper_bound_x(dist: Distribution) -> torch.Tensor:
        """Returns the x upperbound for each of the component distributions.

        If cdf(x) recieves values that are too large, it throws an error. We instead find a suitable upper bound and
        later clip the x values to this range. The upper bound should:

            - Not throw an error when it is used in `cdf(upperbound).`
            - As per "Approximation principles" it should not result in overestimates of the analytical cdf.

        Args:
            dist: The component distribution.

        Return:
            Tensor of shape (dist.batch_shape).

        Details:
            TransformedDistribution are officially bounded btween (finfo.tiny, 1 - finfo.eps). There do not appear to be
            issues using `cdf(x)` for x values corresponding to slighly larger than `1-finfo.eps`. These produce values
            up to 1.

            NOTE: IF we can confirm distribution behaviour is safe outside of the offical bounds specified on the
            distribution (`1-finfo.eps`) this value could be increased. Currently we underestimates any value outside
            `1-finfo.eps` as we are unsure of the behaviour within the range `[1-finfo.eps, 1]`.As per "Approximation
            principles" this is acceptable.
        """
        dtype = dist_dtype(dist)
        finfo = torch.finfo(dtype)
        eps = torch.tensor(finfo.eps, dtype=dtype)

        return dist.icdf(1 - eps)

    def cdf(self, x: torch.Tensor) -> torch.Tensor:
        """Return the CDF.

        Identical to MixtureSameFamily implementation except for clamping.

        Args:
            x: Values to calcuate the CDF for. Must be broadcastable with the `ApproximateMixture.batch_shape`. E.g

                - `self.component_distribution.batch_shape = (2,5)` (last dimension is the components that are combined
                   to make a single Mixture distribution)
                - `self.batch_shape = (2,)
                - `x.shape = (10)` This will fail as it is not broadcastable
                - `x.shape = (10,1)` This will pass as it is broadcastable

        Returns:
            Tensor of shape (x.shape[:-1], self.batch_shape)
        """
        if self.check_dtype and torch.finfo(x.dtype).resolution > torch.finfo(self.dtype).resolution:
            msg = (
                f"Input type {x.dtype} is less precise than dtype of the distribution {self.dtype}."
                " This can lead to loss of precision."
            )
            raise TypeError(msg)

        # Expected x already broadcasts to `self.batch_dimension`.
        # Padding is added so x broadcasts to each of the component distributions that make up the mixture
        x = self._pad(x)
        clipped_x = torch.clamp(x, self.x_bound_lower, self.x_bound_upper)
        cdf_x = self.component_distribution.cdf(clipped_x)
        mix_prob = self.mixture_distribution.probs

        # This is the approach found in torch. Can introduce small numerical errors, but we assume these are neglible.
        return torch.sum(cdf_x * mix_prob, dim=-1)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate the log prob (e.g log(pdf)).

        Args:
            x: Values to calcuate the log prob for. Must be broadcastable with the `ApproximateMixture.batch_shape`. E.g

                - `self.component_distribution.batch_shape = (2,5)` (last dimension is the components that are combined
                   to make a single Mixture distribution)
                - `self.batch_shape = (2,)
                - `x.shape = (10)` This will fail as it is not broadcastable
                - `x.shape = (10,1)` This will pass as it is broadcastable

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
        dist: `(*batch_shape,)` mixture distribution producing events of event_shape samples.
        q: quantile to find the inverse cdf of. Must be boardcastable up to (`*batch_shape`,). Must not have more
           dimensions than `*batch_shape` (only 1 q can be passed to each of the distributions in the batch.)

    Returns:
        tensor of shape (2,*batch_shape), there the first index represents the lower
        bounds, and the second the upper bounds.

    Details:
    Mixture distribution calculate the CDF as follows:

        $q = w_i * CDF_1(y) + w_2 * CDF_2(y) + ... + w_n * CDF_n(y)$
        Which can be written as: $q = w_i * q_1 + w_2 * q_2 + ... + w_n * q_n$

        where $0 <= w_i <= 1$ and $\\sum{w_i} = 1$

    An effective way to bound the x values the can produce y is:

        - take the `icdf(q)` for each distribution. Now have X_n values.
        - `lower_bound = min(X_n)`: at this point the first component distribution has become big enough to produce q.
          As the weights are between [0,1] no point prior would be able to procude q as no q_i was large enough.s
        - `upper_bound = max(X_n)`: at this point the last component distribution has become big enough to produce q.
          As the weights sum to one, q must be produced by this point.
    """
    # expand can not be done to smaller shapes, so this safely handled the case when q.dim() > len(dist.batch_shape)
    q_expanded = q.expand(dist.batch_shape)

    q_expanded = q_expanded.unsqueeze(-1)  # add a dimension for the underlying component distributions.
    values = dist.component_distribution.icdf(q_expanded)
    lower_bound = values.min(dim=-1).values
    upper_bound = values.max(dim=-1).values

    return torch.stack([lower_bound, upper_bound])
