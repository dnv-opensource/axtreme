"""GPBruteForce calculates QoIs by applying the surrogate to each env sample in the period."""
# %%
# pyright: reportUnnecessaryTypeIgnoreComment=false

import warnings
from collections.abc import Iterable
from contextlib import ExitStack
from typing import TypeVar

import gpytorch
import torch
from botorch.models.model import Model
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from torch.distributions import Gumbel
from torch.nn.modules import Module

from axtreme.qoi.qoi_estimator import QoIEstimator
from axtreme.sampling import MeanVarPosteriorSampledEstimates, NormalIndependentSampler, PosteriorSampler

T = TypeVar("T")


class GPBruteForce(MeanVarPosteriorSampledEstimates, QoIEstimator):
    """Estimate the QoI for an extreme response distribution, using a surrogate model.

    Uses a full periods of environment samples, and passes each sample through the surrogate.

    Overview of the algorithem:

    - Take `n` periods of environment data points.
    - Use the surrogate model to estimate the likely response distribution at each point (the posterior).
    - Take `n_posterior_samples` of the posterior, each representing a guess at what the true simulator is.

    - For each posterior sample:

      - Simulate the response seen at each data point.
      - Find the largest response in each period. Each of these is a sample of that posterior's ERD.
      - Calculate the QoI based on these ERD samples.

    - Return the QoIs calculated by each posterior sample.

    Uncertainty in the results comes from three sources:

    - The envirionment samples used.
    - Uncertainty in the GP and the posterior samples used.
    - Randomness from sampling the surrogates output distribution.

    Optimisation Notes:
        GPBruteForce is not smooth w.r.t to smooth changes in the model (e.g like provided By QoILookAhead).

    Todo:
        Provide reference to the gpbruteforce.md file in docs so it renders (sw 2024-11-29).
    """

    def __init__(  # noqa: PLR0913
        self,
        env_iterable: Iterable[torch.Tensor],
        input_transform: InputTransform | None = None,
        outcome_transform: OutcomeTransform | None = None,
        posterior_sampler: PosteriorSampler | None = None,
        *,
        erd_samples_per_period: int = 1,
        shared_surrogate_base_samples: bool = False,
        device: torch.device | None = None,
        no_grad: bool = True,
        seed: int | None = None,
    ) -> None:
        """Initialise the QOI estimator.

        Args:
            env_iterable: An iterable that produces the env data to be used. Typically this is a DataLoader.

                - The iterable contains batches of shape (n_periods, batch_size, d).
                - Combining all of the batch should produce the shape (n_periods, period_len, d).
                - This is an iterable because predictions often need to be made in batches for memory reasons.
                - If your data is small, you can process it all at once by passing `[data]`, where `data` is a tensor.

            input_transform: Transforms that should be applied to the env_samples before being passed to the model.
            outcome_transform: Transforms that should be applied to the output of the model before they are used.
            posterior_sampler: The sampler to use to draw samples from the posterior of the GP.

                - `n_posterior_samples` is set in the PosteriorSampler.

                .. Note::
                    If `env_iterable` contains batches, a batch-compatible sampler, such as
                    `NormalIndependentSampler`, should be chosen.

            erd_samples_per_period: Number of ERD samples created from a single period of data. This can reduce the
                noise of sampling the response drawn from the surrogate's response distribution (at a point 'x').
            shared_surrogate_base_samples: If True, all `n_posterior_samples` will use the same base samples when
                sampling the surrogate's response output. As a result, the posterior samples are responsible for any
                difference in ERD distribution (e.g., surrogate sampling noise no longer contributes).

                - Set to False: Better shows overall uncertainty in QoI.
                - Set to True: Shows only uncertainty caused by GP uncertainty.

            device: The device that the model should be run on.
            no_grad: Whether to disable gradient tracking for this QOI calculation.
            seed: The seed to use for the random number generator. If None, no seed is set.
        """
        super().__init__()
        self.env_iterable = env_iterable
        self.input_transform: InputTransform | None = input_transform
        self.outcome_transform: OutcomeTransform | None = outcome_transform
        self.posterior_sampler: PosteriorSampler = posterior_sampler or NormalIndependentSampler(torch.Size([100]))
        self.erd_samples_per_period: int = erd_samples_per_period
        self.shared_surrogate_base_samples = shared_surrogate_base_samples
        self.device: torch.device = device or torch.device("cpu")
        if self.device != torch.device("cpu"):
            msg = "GPU calcualtion are not yet supported, torch.device('cpu') must be used."
            raise NotImplementedError(msg)
        self.no_grad: bool = no_grad
        if not self.no_grad:
            msg = "Gradient tracking is not yet supported, please set `no_grad=True`."
            raise NotImplementedError(msg)
        self.seed: int | None = seed

    @staticmethod
    def sample_surrogate(
        params: torch.Tensor, n_samples: int = 1, base_sample_broadcast_dims: list[int] | None = None
    ) -> torch.Tensor:
        """Create the surrogate model for a given set of input parameters, and sample response of the surrogate.

        Typically a GP is used to parameterise the surrogate model at a specific x. The now parameterise model can be
        run multiple times to get different realisations of the stochastic response.

        Args:
            params: (`*b`, p) tensor of parameters. The last dimesion is expected to contain the parameters required to
            instantiate a single surrogate model. All other dimensions are optional batch dimension.
            n_samples: The number of samples to draw from the surrogate model at a single x point.
            base_sample_broadcast_dims: List of indexes in (`*b`). Base samples will be shared (broadcast) across these
                dimension of `*b`. For example:

                - params.shape is (n_posterior_samples, n_periods, batch_size, n_params).

                  - `*b` = (n_posterior_samples, n_periods, batch_size)
                  - p = (n_params)

                - You would like to use the same base samples for each n_posterior_samples, so that any difference in
                  output can be attributed to the difference in the n_params, rather than due to the randomness in the
                  sample generated by the surrogate mode.

                - By setting `base_sample_broadcast_dims=[0]` the base samples used would be of shape
                  (1, n_periods, batch_size), which would achieve the desired effect.

        Returns:
            tensor of (n_samples, `*b`) representing the response of the surrogate model.

        Todo:
            The `base_sample_broadcast_dims` behaviour is challenging to describe now that is in this function rather
            than in context. Alternately base samples could be directly applied like in
            posterior.rsample_from_base_samples. We have avoid this so complexity is contained here for now.
        """
        gumbel_dists = Gumbel(
            params[..., 0],
            params[..., 1].clamp(min=0.0000001),
            validate_args=True,
        )

        b_shape = list(params.shape[:-1])
        # Change dimension so they will be broadcast if required.
        if base_sample_broadcast_dims:
            for idx in base_sample_broadcast_dims:
                b_shape[idx] = 1

        base_samples_shape = torch.Size([n_samples]) + tuple(b_shape)

        # This is a home baked version of gumbel_dists.rsample_from_base_samples
        # broadly speaking it produces the same results as gumbel_dists.rsample(n_samples)
        base_samples = torch.rand(base_samples_shape)
        gumbel_samples = gumbel_dists.icdf(base_samples)

        return gumbel_samples

    def _process_batch(self, env_batch: torch.Tensor, model: Model) -> torch.Tensor:
        """Process batches of env_data, returning the max response seen across each period.

        For each period, n_posterior_samples provide different plausable estimates of the response distribution.
        erd_samples_per_period are then drawn from each response distribution. Each of these is stored seperately so
        they can later be aggregated across batches.

        Args:
            env_batch: (n_periods, batch_size, d)
            model: The gp to be used as the surrogate model

        Return:
            (erd_samples_per_period, n_posterior_samples, n_periods)
        """
        if self.input_transform:
            env_batch = self.input_transform.transform(env_batch)

        posterior = model.posterior(
            env_batch,
            posterior_transform=self.outcome_transform.untransform_posterior if self.outcome_transform else None,
        )

        # Sampling from the posterior using the provided sampler
        # shape is now: (n_posterior_samples, n_periods, batch_size, n_targets)
        gp_samples: torch.Tensor = self.posterior_sampler(posterior)

        # If using botorch based samplers this is an attribute we can access, but we use a less specific interface
        dim_posterior_samples = len(gp_samples.shape) - len(posterior.mean.shape)  # pyright: ignore[reportAttributeAccessIssue]

        # shape is now: (erd_samples_per_period, n_posterior_samples, n_periods, batch_size)
        surrogate_responses = self.sample_surrogate(
            gp_samples,
            self.erd_samples_per_period,
            base_sample_broadcast_dims=list(range(dim_posterior_samples))
            if self.shared_surrogate_base_samples
            else None,
        )

        # For each time period, find the max response
        # Shape is now:  (erd_samples_per_period, n_posterior_samples, n_periods)
        extreme_responses = surrogate_responses.max(dim=-1).values  # noqa: PD011

        return extreme_responses

    def posterior_samples_erd_samples(self, model: Model) -> torch.Tensor:
        """Returns the erd samples created by each posterior sample.

        __call__ uses these erd sample to create a QoI estimate per posterior.

        Args:
            model: The GP model to use for the QOI estimation. It should have output dimension 2 which represents the
                location and scale of a Gumbel distribution.

        Returns:
            Tensor: The erd samples obtained for each function (posterior sample) obtianed from the GP.
            Shape: (n_posterior_samples, n_periods * erd_samples_per_period)
        """
        with ExitStack() as stack:
            # Adding context to the stack according to the configuration
            if self.no_grad:
                stack.enter_context(torch.no_grad())
            if self.seed is not None:
                stack.enter_context(torch.random.fork_rng())

                # If a seed is provided, set the seed for the random number generator
                # This is done in the context manager to ensure that the seed is only set for this calculation
                _ = torch.manual_seed(self.seed)
            stack.enter_context(gpytorch.settings.fast_pred_var())

            # make sure transforms are on the right device.
            if self.input_transform:
                assert isinstance(self.input_transform, Module)
                self.input_transform = self.input_transform.to(self.device)
            if self.outcome_transform:
                self.outcome_transform = self.outcome_transform.to(self.device)

            # This will track the most extreme value seen thus far in the iterations
            extreme_value_thus_far: torch.Tensor | None = None
            for batch in self.env_iterable:
                # This is of shape: (n_periods, batch_size, d)
                env_batch = batch.to(self.device)

                # This has shape: (erd_samples_per_period, n_posterior_samples, n_periods)
                batch_extreme_responses = self._process_batch(env_batch, model)

                # If a bigger extreme value has been seen, replace the tracked value
                extreme_value_thus_far = (
                    extreme_value_thus_far.max(batch_extreme_responses)
                    if extreme_value_thus_far is not None
                    else batch_extreme_responses
                )

            if extreme_value_thus_far is None:
                msg = "env_iterable provided was empty"
                raise ValueError(msg)

            # Group the ERD samples that belong to the same posterior sample
            # Shape is now:  (n_posterior_samples, n_periods * erd_samples_per_period)
            extreme_value_thus_far = torch.cat([*extreme_value_thus_far], dim=-1)
            return extreme_value_thus_far

    def __call__(self, model: Model) -> torch.Tensor:
        """Estimate the QOI using the given GP model.

        Args:
            model: The GP model to use for the QOI estimation. It should have output dimension 2 which represents the
                location and scale of a Gumbel distribution.

        Returns:
            Tensor: 1d tensor where each entry is an estimate of the QOI as produced by function sampled from the GP.
            The function sampled from the GP (and the number of them) is determined by `posterior_sampler`.
        """
        posterior_samples_erd_samples = self.posterior_samples_erd_samples(model)

        n_erd_samples = posterior_samples_erd_samples.shape[-1]
        if n_erd_samples % 2 == 0:
            msg = (
                f"There are {n_erd_samples} erd samples for each sample frrom the GP (posterior samples)."
                " tensor.median() called on a even number of samples produces a systematic underestimate of the median"
                " (because `torch.tensor([1,2,3,4]).median() -> 2`). n_erd_samples is determined by n_periods"
                " * erd_samples_per_period (where n_periods=env_iterable.shape[-3])"
            )
            warnings.warn(msg, UserWarning, stacklevel=2)

        # TODO(ks): This should maybe be changed to include different statistic than the median (any quantile?)
        # Would need to update the warning when changing this
        return posterior_samples_erd_samples.median(dim=-1).values  # noqa: PD011
