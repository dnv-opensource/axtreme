"""Defines the interface posterior samplers are expected to adhear to."""

from collections.abc import Callable
from typing import Protocol

import torch
from botorch.posteriors import GPyTorchPosterior


class PosteriorSampler(Protocol):
    """Defines the protocol for sampler function.

    This follows the definition of the MCSampler.forward() method, but allows for simple function based implementations.

    Note:
        Function using posteriors should check if there are special `mean` and `var` methods if they support posterior
        samplers that require them (e.g UT). See note below for details.

    Notes on samplers the require special aggregation of results:

    - We use a surrogate model to estimate a Quantity of Interest (QOI), and provide uncertainty of that estimate.
      Typically it is challenging to calculate this on the posterior directly, so often we take samples of the
      posterior and calculate the value of interest using those. We can then combine those estimates to give the QOI
      mean/variance for the overall posterior.
    - Some methods (e.g UT) select posterior samples in a special way which means less samples are needed. The
      calculations for each of these samples then need to be combined in a special way to estimate the QOI
      mean/variance for the overall posterior. If such special methods are needed, they should be implement on the
      posterior sampler with the following signature.

      .. code-block:: python

        def mean(self, x: torch.Tensor) -> torch.Tensor: ...
        def var(self, x: torch.Tensor) -> torch.Tensor: ...

      Where

      - `x` is the scores (e.g QOIs) for each posterior sample.

        - It is expected to be of shape (*batch_shape*)

        .. Todo::
                Batching gets confusing here, how should we do UT of multiple batch dims? Is it only
                ever flat, should we say that here?

        - Output is a scalar.

    Todo:
        Need to fix/revisit this. Feel wrong that functions that use posterior need to check for some optional methods
        that "might" be there (and those functions are not programmatically defined as part of the protocol). Extra
        info in #132.

        .. Note::
            The functions are not on the protocol because we want MCSampler from botorch to fall into protocol
            definition.
    """

    def __call__(self, posterior: GPyTorchPosterior) -> torch.Tensor:
        """Produces samples from a posterior distibution.

        Args:
            posterior: the posterior object to be sampled

        Return:
            Tensor: shape (`*sample_shape, *b, n, m`), where:

              - `sample_shape`: the number/shape of samples produces
              - `*b, n, m`: as defined in glossary.

        """
        ...


class MeanVarPosteriorSampledEstimates(Protocol):
    """This is a protocol for classes that computes estimates using a PosteriorSampler.

    Some posterior samplers require that mean and variance of the estimates calculated by sampling from it need to be
    calculated in a special way. This protocol allows for these special methods to be easier to be used on the
    outside of the class that inherits from this protocol. One example of a case where this is useful is when using a
    posterior sampler in a QoIEstimator. For more information check issue #132.

    Example of usage:

    .. code-block:: python

        class MyClass(MeanVarPosteriorSampledEstimates):
            def __init__(self, posterior_sampler: PosteriorSampler):
                self.posterior_sampler = posterior_sampler

            def f(self, x: torch.Tensor) -> torch.Tensor:
                # Some functions that takes the samples and returns a tensor
                # The dimension representing the posterior_sampling should be preserved.
                ...

            def foo(self) -> torch.Tensor:
                #  Get a model and points x to evaluate posterior
                x: torch.Tensor = ...
                model: botorch.models.Model = ...

                # Calculating posterior
                posterior = model.posterior(x)

                # Sampling from the posterior
                samples = self.posterior_sampler(posterior)

                return self.f(samples)


        estimator = MyClass(posterior_sampler)
        result = estimator.foo()

        # Using the mean and var methods from MeanVarPosteriorSampledEstimates
        # This will use the mean and var methods from the posterior_sampler if they are available.
        mean = estimator.mean(result)
        var = estimator.var(result)
    """

    posterior_sampler: PosteriorSampler

    def mean(self, x: torch.Tensor) -> torch.Tensor:
        """Function that computes the mean of the estimates produced by using `self.posterior_sampler`.

        The dimension representing the posterior_sampling should be preserved from the sampling using
        `self.posterior_sampler` to sample a posterior until using this method.

        For many applications this method will just be using a default implementation that computes the mean.
        E.g using torch.mean(x) like in this implementation.

        However, in some special cases, it might be useful to provide a custom implementation
        to give a more accurate estimate.
        For instance if one uses UTSampler to sample the posterior
        the mean should be estimated as a weighted sum of the estimates.

        This function uses the mean method of the posterior_sampler if it is available,
        otherwise it uses the default implementation.

        Args:
            x: A tensor of the estimates with shape (num_posterior_samples,).

        Returns:
            torch.Tensor: The mean of the the estimates. Should be a scalar with shape ().
        """
        # If there is a `mean()` method in the `posterior_sampler` then use it.
        # Otherwise use the default implementation from `torch`.
        mean_func: Callable[[torch.Tensor], torch.Tensor] | None = getattr(self.posterior_sampler, "mean", None)
        return mean_func(x) if mean_func and callable(mean_func) else torch.mean(x, dim=0)

    def var(self, x: torch.Tensor) -> torch.Tensor:
        """Function that computes the variance of the estimates produced by using `self.posterior_sampler`.

        The dimension representing the posterior_sampling should be preserved from the sampling using
        `self.posterior_sampler` to sample a posterior until using this method.

        For many applications this method will just be using a default implementation that computes the variance.
        E.g using torch.var(x) like in this implementation.

        However, in some special cases, it might be useful to provide a custom implementation
        to give a more accurate estimate.
        For instance if one uses UTSampler to sample the posterior
        the variance should be estimated in a special way.

        This function uses the variance method of the posterior_sampler if it is available,
        otherwise it uses the default implementation.

        Args:
            x: A tensor of the estimates with shape (num_posterior_samples,).

        Returns:
            torch.Tensor: The variance of the the estimates. Should be a scalar with shape ().
        """
        # If there is a `var()` method in the `posterior_sampler` then use it.
        # Otherwise use the default implementation from `torch`.
        var_func: Callable[[torch.Tensor], torch.Tensor] | None = getattr(self.posterior_sampler, "var", None)
        return var_func(x) if var_func and callable(var_func) else torch.var(x, dim=0)
