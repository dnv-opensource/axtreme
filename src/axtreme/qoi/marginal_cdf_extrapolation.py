"""MarginalCDFExtrapolation calculates QoIs by estimating the behaviour of a single timestep, and extrapolating."""

# pyright: reportUnnecessaryTypeIgnoreComment=false
# %%
import warnings
from collections.abc import Iterable
from contextlib import ExitStack

import gpytorch
import torch
from botorch.models.model import Model
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from torch.distributions import Categorical, Distribution, Gumbel
from torch.nn.modules import Module

from axtreme.distributions import icdf
from axtreme.distributions.mixture import ApproximateMixture, icdf_value_bounds
from axtreme.qoi.qoi_estimator import QoIEstimator
from axtreme.sampling import MeanVarPosteriorSampledEstimates, PosteriorSampler, UTSampler


class MarginalCDFExtrapolation(MeanVarPosteriorSampledEstimates, QoIEstimator):
    """Estimate an ERD QoI using the marginal CDF for a single timestep.

    Basic Idea:
        - Get the marginal CDF of the response for a single timestep (e.g. 1 hour).
            Marginal CDF: The CDF after marginalising out all sources of randomness within a timestep. E.g if you
            considered all the variablity in a single timestep (e.g all the different weather conditions), collected
            the different CDFs they produce, and then averaged them, you would have the marginal CDF.
        - Using the average CDF you can then calculate:
            - "The probablity that a response of size `y` won't be exceeded in 1 hour" as follows:
                - :math:`CDF(y = .8) = .5`
            - "The probablity that a response of size `y` won't be exceeded in 2 hours" as follows:
                - ``[Prb not exceeded in hour one]`` AND ``[Prb not exceeded in hour two]``
                - :math:`CDF(y = .8) * CDF(y = .8) = .5 * .5 = .25`
            - "The probablity that a response of size `y` won't be exceeded in N hours" as follows:
                - :math:`CDF(y = .8)^N = (.5)^N`

        This is possible because we are using the 'average' timestep, which is the same for all timesteps.

    Strengths:
        - Once the 'average' CDF has been obtained, very large values of N can be calculated quickly.

    Challenge:
        - Need to obtain the 'average' CDF, and it must be very accurate.

    See [#TODO(sw 2024_11_4) put in link to pre-print] for details.
    """

    def __init__(  # noqa: PLR0913
        self,
        env_iterable: Iterable[torch.Tensor] | Iterable[list[torch.Tensor]],
        period_len: int,
        quantile: torch.Tensor | None = None,
        input_transform: InputTransform | None = None,
        outcome_transform: OutcomeTransform | None = None,
        posterior_sampler: PosteriorSampler | None = None,
        response_distribution: type[Distribution] = Gumbel,
        quantile_accuracy: torch.Tensor | None = None,
        dtype: torch.dtype = torch.float64,
        *,
        device: torch.device | None = None,
        no_grad: bool = True,
    ) -> None:
        """Initialise the QOI estimator.

        Args:
            env_iterable: An interable that produces the env data to be used. Typically this is a DataLoader.
                Supports standard env data, and weighted env data (e.g Importance sampled).

                - Standards env data: Expects tensor of shape (batch_size, d)
                - Weighted data: Expectes list of tensors.

                    - The first: the env data of shape (batch_size,d).
                    - The second item: the weight, of shape (batch_size,).

            period_len: The number of env samples in the period of interest.
            quantile: Shape (,). The quantile of the ERD to be estimate (the QoI). Should be within range (0,1).
              default .5.
            input_transform: transforms that should be applied to the env_samples before being passed to the model
            outcome_transform: transforms that should be applied to the output of the model before they are used
            posterior_sampler: The sampler to use to draw samples from the posterior of the GP.

                - ``n_posterior_samples``: is set in the PosteriorSampler
                - NOTE: if env_iterable contains batches, a batch compatible sampler such as
                  ``NormalIndependentSampler`` or "ut_sampler" should be selected.

            response_distribution: The distribution which models the stochastic response of the simulation at a single
              input point.
            quantile_accuracy: shape (,). Default value .01. Internally, optimisation is used to find the ``quantile``.
              The optimiser is allowed to terminate once in the region  ``quantile +/- quantile_accuracy``. The greater
              the accuracy required, the longer the optimisation will take. Typically other sources of uncertainty
              produce far greater uncertainty.
            dtype: The dtype used for the distribution. This has implications for the numerical accuracy possible.
            device: The device QoI should be performed on.
            no_grad: If gradient requires tracking through this QOI calculation.
        """
        self.quantile: torch.Tensor = quantile or torch.tensor(0.5, dtype=dtype)
        self.env_iterable: Iterable[torch.Tensor] | Iterable[list[torch.Tensor]] = env_iterable
        self.period_len: int = period_len
        self.input_transform: InputTransform | None = input_transform
        self.outcome_transform: OutcomeTransform | None = outcome_transform
        self.posterior_sampler: PosteriorSampler = posterior_sampler or UTSampler()
        self.response_distribution: type[Distribution] = response_distribution
        self.quantile_accuracy: torch.Tensor = quantile_accuracy or torch.tensor(0.01, dtype=dtype)
        self.dtype = dtype
        self.device: torch.device = device or torch.device("cpu")
        if self.device != torch.device("cpu"):
            msg = "GPU calcualtion are not yet supported, torch.device('cpu') must be used."
            raise NotImplementedError(msg)
        self.no_grad: bool = no_grad
        if not self.no_grad:
            msg = "Gradient tracking is not yet supported, please set `no_grad=True`."
            raise NotImplementedError(msg)

    def __call__(self, model: Model) -> torch.Tensor:
        """Estimate the QOI using the given GP model.

        Args:
            model: The GP model to use for the QOI estimation. It should have output dimension 2 which represents the
                location and scale of a Gumbel distribution.

        Returns:
            torch.Tensor: A tensor, with shape (n_posterior_samples,), of the estimated QoI for each of the
            functions sampled from the GP using the posterior sampler.
        """
        # resulting shape is (n_posterior_samples, n_env_samples, n_targets) and _posterior_samples, n_env_samples)
        params, weigths = self._parameter_estimates(model)

        if params.dtype != self.dtype or weigths.dtype != self.dtype:
            msg = (
                f"The model produced parameters of dtype {params.dtype} and weigths of dtype {weigths.dtype}."
                f" This does not match the specified dtype {self.dtype}."
                " Parameters and weights will be converted to this dtype to create a suitable distribution."
                " NOTE: underlying parameters are still only accurate to float32 precision, this may effect overall"
                " accuracy. Recommended make float64 preicitons with the model."
            )
            warnings.warn(msg, stacklevel=8)

            params, weigths = params.type(self.dtype), weigths.type(self.dtype)

        # Create the the marginal distribution for the parameters. The categorical distribution handles weights scaling
        # resulting batch shape (n_posterior_samples)
        if self.response_distribution is Gumbel:
            assert issubclass(self.response_distribution, Gumbel)  # help mypy determine the type
            loc = params[..., 0]
            scale = params[..., 1].clamp(min=1e-6)  # Ensure scale is positive
            component_dist = self.response_distribution(loc, scale)
        else:
            # To support other distribution need to to apply parameter constraints (list on the distribution), to the
            # the input params. Additionally need to know the order (name) of params in params.
            raise NotImplementedError("Only Gumbel distributions are currently supported.")

        # ApproximateMixture is appropriate becuase it produces conservative estimate. The optimisation method
        # later check that the dist has suitable resolution for the required accuracy.
        dist = ApproximateMixture(
            mixture_distribution=Categorical(weigths),
            component_distribution=component_dist,
        )

        # determine the timestep quantile to be found, and the acceptable margin of error
        _qtimestep = q_to_qtimestep(self.quantile.item(), self.period_len)
        max_qtimestep_error = acceptable_timestep_error(self.quantile.item(), self.period_len)

        # calculated the icdf values
        qtimestep = torch.tensor(_qtimestep, dtype=torch.float64)
        opt_bounds = icdf_value_bounds(dist, qtimestep)
        qois = icdf.icdf(dist, qtimestep, max_qtimestep_error, opt_bounds)
        # TODO(sw 2024-12-16): Gradient should be reattached here.
        return qois

    def _parameter_estimates(self, model: Model) -> tuple[torch.Tensor, torch.Tensor]:
        """Estimate the parameters of the individual distributions the make up the marginal distribution.

        Estimate the parameters of the component distribtuions (individual distribution that will later be marginalised)
        , and the associated mixure weights (importance weight of each of the samples in the marginalisation).

        Args:
            model: The GP model to use for the QOI estimation. The number of output distributions should match that
            required by ``response_distribution``.

        Returns:
            Tuple of tensors where:
                - Item 1: The parameters of the component distributions.
                  Shape (n_posterior_samples, n_env_samples, n_targets)
                - Item 2: The probability (importance weight) to associated with each of the component distributions.
                  Shape (n_posterior_samples, n_env_samples). The same weights are used across all posterior samples.
                  These weights are expected to be non-negative.
        """
        with ExitStack() as stack:
            # Adding context to the stack according to the configuration
            if self.no_grad:
                stack.enter_context(torch.no_grad())
            # Unclear the impact of this sett, but accoring the this shouldn't have downside. https://arxiv.org/abs/1803.06058
            stack.enter_context(gpytorch.settings.fast_pred_var())

            # make sure transforms are on the right device.
            if self.input_transform:
                assert isinstance(self.input_transform, Module)
                self.input_transform = self.input_transform.to(self.device)
            if self.outcome_transform:
                self.outcome_transform = self.outcome_transform.to(self.device)

            batched_posterior_samples_list: list[torch.Tensor] = []
            importance_weights_list: list[torch.Tensor] = []

            """NOTE: We can't easily batch the full calculation, so we just batch the posterior prediction. This can
            have runtime benefits when making the posterior predictions, and means users can provide env data in a
            format similar to other QoI methods. We then store the full dataset of importance weights and prediction,
            but this is not anticipated to be a problem, as this method does not need to use a large amount of
            ``env_samples``. Will revist this if it proves to be an issue."""
            # TODO(sw 2024-12-16): If getting too large, batch on posterior samples (make the sampler a generator?)
            for env_batch in self.env_iterable:
                # If items are lists, this is the weights are being provided (e.g Importance Sampling case).
                if isinstance(env_batch, list):
                    # The shapes should be (batch_size, d) and (batch_size,) respectively.
                    env_samples, importance_weights = env_batch
                    importance_weights_list.append(importance_weights)
                else:
                    env_samples = env_batch

                # Making sure the env_batch is on the right device
                env_samples = env_samples.to(self.device)

                if self.input_transform:
                    env_samples = self.input_transform.transform(env_samples)

                posterior = model.posterior(
                    env_samples,
                    posterior_transform=self.outcome_transform.untransform_posterior
                    if self.outcome_transform
                    else None,
                )
                # Shape is now: (n_posterior_samples, batch_size, n_targets)
                posterior_samples_batch = self.posterior_sampler(posterior)
                batched_posterior_samples_list.append(posterior_samples_batch)

            # Combines posterior samples and importance weights from all the batches
            # This is now of shape (n_posterior_samples, n_env_samples, n_targets)
            posterior_samples = torch.cat(batched_posterior_samples_list, dim=-2)

            # use importance weights if provided, otherwise treat all as equally weighted
            if importance_weights_list:
                importance_weights = torch.cat(importance_weights_list, dim=0).to(self.device)
                if importance_weights.min() < 0:
                    raise ValueError("Importance weights must be non-negative.")
            else:
                importance_weights = torch.ones(1, device=self.device, dtype=posterior_samples.dtype)

            # Provide a weight for each posterior distributionl.
            importance_weights = importance_weights.expand(posterior_samples.shape[:-1])

            return posterior_samples, importance_weights


def q_to_qtimestep(q: float, period_len: int) -> float:
    """Convert a long term quantile to an equivalent single timestep quantile.

    Args:
        q: long term quantile being estimated.
        period_len: the number of timesteps that make up the period of interest.

    Returns:
        The equivalent quantile for a single timestep with error +-

    Note:
        This funciton exists because there were numerical concerns for this process. Having a function allows us to
        document them in tests. It is appropriately accuract for very large periods. Periods of 1e13 creates an error of
        less that 1e-3 in q (the full quantile estimate). See
        `test/qoi/test_margianl_cdf_extpolation/test_q_to_qtimestep_numerical_precision_period_increase` for details.
    """
    return q ** (1 / period_len)


def acceptable_timestep_error(q: float, period_len: int, atol: float = 0.01) -> float:
    """Maximum possible single timestep error while still producing required accuracy in period estimate.

    We often make estimates for a single timestep, and scale them to a period (many timesteps). e.g.
    `q = q_timestep^period_len`. Errors in the single timestep estimates compound. This function returns the largest
    error possible to a single timestep to still be within the required accuracy in the period estimate.

    Args:
        q: long term quantile being estimated.
        period_len: the number of timesteps that make up the period of interest.
        atol: the maximum absolute error acceptable in q

    Returns:
        The maximum absolute error acceptable in the q_timestep estimate.
    """
    # NOTE: We only look at `q + atol` because this will produce a larger error than `q - atol`
    # (because of the exponential relationship)
    acceptable_error = q_to_qtimestep(q + atol, period_len) - q_to_qtimestep(q, period_len)

    FLOAT64_RESOLUTION = 1e-15  # noqa: N806
    if acceptable_error < (FLOAT64_RESOLUTION * 10):
        msg = (
            f"The acceptable error is {acceptable_error}. This is approaching or below the limits of floating point 64"
            " precision (1e-15)."
        )
        raise ValueError(msg)
    return acceptable_error
