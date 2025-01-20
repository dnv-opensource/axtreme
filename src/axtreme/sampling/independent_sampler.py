"""Base class for sampling methods that ignore covariance between different input points.

Todo: TODO
- For MultiTask models (e.g GPs with multiple targets that are not indpendant), this covariance should be considered.
- Currently ALL covariance is ignored.
"""

from abc import ABC, abstractmethod

import linear_operator
import torch
from botorch.posteriors import GPyTorchPosterior, Posterior
from botorch.sampling import MCSampler
from gpytorch.distributions.multitask_multivariate_normal import MultitaskMultivariateNormal
from gpytorch.distributions.multivariate_normal import MultivariateNormal


class IndependentMCSampler(MCSampler, ABC):
    r"""Abstract base class for MCSamplers that independantly apply the same set of base samples at each x point.

    This should be used when you want to ignore the covariance between different x points, and sample the X output
    space independently.

    Note:
        Dimensions of the posterior are described using botorch notation (detailed in glassary.md) where:

        - \*b: batch shape (can have arbitrary dimensionality).
        - n: the number of x input points.
        - m: dimensionality of targets space.

        For example:

        - MultivariateNormal: (\*b, n)
        - MultitaskMultivariateNormal: (\*b, n, m)

    *Dev Notes*:
      - Is the shape of the base samples up to us to define, or are there some other parts of the system that expect
        MCSampler to have certain shape.

          Answer: MCSampler does not define base_sample - we can do what we want.

      - MultivariateNormal and MultitaskMultivariateNormal both requires shape (\*sample_shape, \*posterior.shape()
        for `base_samples` in `posterior.rsample_from_base_samples`

        - For MultitaskMultivariateNormal: posterior.shape() = (\*b,n,m)
        - For MultivariateNormal: posterior.shape() = (\*b,n)

    **Design decision**: we decide to make m explicit throughout.

      - e.g base samples will always have shape (\*sample_shape,m)
      - This makes our code cleaner. When then convert back to implicit m dimension when calling
        `posterior.rsample_from_base_samples`
    """

    def forward(self, posterior: Posterior) -> torch.Tensor:
        r"""Draw samples from the posterior, treating each of the n input points as independant.

        Args:
          posterior: The posterior which should be sampled.

        Returns:
          Returns a samples of the posterior with shape: (\*sample_shape, \*b, n, m) where:
          - \*b: is the posterior batch shape
          - n: number of input points in posterior
          - m: dimensionality of target

          NOTE: While MultitaskMultivariateNormal and MultivariateNormal have different shapes ((\*b,n,m) and (\*b,n))
          the output of this function is always of shape (\*sample_shape, \*b, n, m).
          This is consistent with the behaviour of `MCSampler`.
        """
        if not isinstance(posterior, GPyTorchPosterior):
            msg = f"Currently only supports posterior of type GPyTorchPosterior. Got {type(posterior)}"
            raise NotImplementedError(msg)

        # This removes covariance from the posterior because when `posterior.rsample_from_base_samples` is applied the
        # base samples are passed through the tranformation: \mu + L*base_samples. Where
        #   - \mu: is the mean
        #   - L: is the "root" of the convariance matrix.
        # If there is covariance in L, independant base_samples will produce covariate results
        edited_posterior = diagonalize_distribution(posterior)

        # Produces base_samples of size (*sample_shape, m)
        self._construct_base_samples(edited_posterior)

        extended_base_samples = self._extend_base_samples(self.base_samples, posterior)

        # required base sample shape:
        #  - (*b,n,m) for MultitaskMultivariateNormal
        #  - (*b,n)  for MultivariateNormal
        return edited_posterior.rsample_from_base_samples(
            sample_shape=self.sample_shape, base_samples=extended_base_samples
        )

    @abstractmethod
    def _construct_base_samples(self, posterior: GPyTorchPosterior) -> None:
        """Build the base samples that should be used for sampling the GP output produced by a single input (x).

        `IndependentMCSampler`s apply the same base samples to every x in the posterior (independantly). The base
        samples to be used are define in this method.

        Return:
            tensor of shape (*sample_shape, m).
            NOTE m dimension should be present even if posterior.distribution is MultivariateNormal.

        This is then store in self.register_buffer("base_samples", base_samples)
        """
        raise NotImplementedError("Subclasses must implement this method")

    @classmethod
    def _extend_base_samples(cls, base_samples: torch.Tensor, posterior: GPyTorchPosterior) -> torch.Tensor:
        """Extend base samples to the shape required for `posterior.rsample_from_base_samples`.

        Args:
            base_samples: of shape (*sample_shape, m)
            posterior: The distibution where the base samples are intended to be applied.

        Return:
            A view of base_smaples of shape (*sample_shape, *posterior.shape()).
                - For MultitaskMultivariateNormal: (*sample_shape, *b, n, m)
                - For MultivariateNormal: (*sample_shape, *b, n)
        """
        # In order to handle the different shaped posterior more easily, we convert them both to (*b,n,m).
        # MultivariateNormal has shape (*b,n) (with an implicit m=1). We convert result back to implicit m=1 at the end.
        explicit_posterior_shape = cls._explicit_posterior_shape(posterior)

        # We eventually want to broadcast this to each n (*sample_shape, *b, n, m)
        # By turning base samples into the following shape it becomes possible: (*sample_shape, *1, 1, m)
        batchable_shape = (
            base_samples.shape[:-1]  # the sample_shape
            + torch.Size([1] * len(explicit_posterior_shape[:-1]))  # *b and n dimension of posterior
            + explicit_posterior_shape[-1:]  # m dimension
        )

        # As all the dimensions being added are 1, the underlying sample points don't change.
        batchable_base_samples = base_samples.view(batchable_shape)

        # build the explicity shape of (*sample_shape, *b, n, m)
        expanded_explicit_base_sample = batchable_base_samples.expand(
            base_samples.shape[:-1] + explicit_posterior_shape
        )

        # posterior.rsample_from_base_samples require an implicity m dim for MultivariateNormal. Squash last dimension.
        if type(posterior.distribution) is MultivariateNormal:
            return expanded_explicit_base_sample.squeeze(-1)

        return expanded_explicit_base_sample

    # Keeping this as a static function as it useful to child methods.
    @classmethod
    def _explicit_posterior_shape(cls, posterior: GPyTorchPosterior) -> torch.Size:
        """Helper methods to return the posterior shape in format (*b,n,m).

        Different posterior distributions have different number of dimensions:
        - MultivariateNormal has shape (*b,n) (with an implicit m=1)
        - MultitaskMultivariateNormal has shape (*b,n,m)

        This is often cumbersom to work with. Instead we work with the more useful format (*b,n,m).

        Args:
            posterior: posterior to determine the explicit shape of.

        Return:
            torch.Size: for shape (*b,n,m)
        """
        dist_type = type(posterior.distribution)
        if dist_type == MultivariateNormal:
            return posterior.shape() + torch.Size([1])
        if dist_type == MultitaskMultivariateNormal:
            return posterior.shape()

        # We do not expect different distibutions, but the ax/botorch codebase is large, so making this explicity.
        msg = (
            f"Expected posterior.distribution of type MultivariateNormal or MultitaskMultivariateNormal.Got {dist_type}"
        )
        raise ValueError(msg)


def diagonalize_distribution(posterior: GPyTorchPosterior) -> GPyTorchPosterior:
    """Diagonalize the distribution of the posterior.

    The points in the posterior need to be treated as independent from each other.
    Therefore we want the covariance matrix to be diagonal.

    Args:
        posterior: The posterior to diagonalize.

    Returns:
        The diagonalized posterior.
    """
    mean = posterior.distribution.mean
    # shape is: (*b,n*m diagonal)
    var = posterior.covariance_matrix.diagonal(dim1=-2, dim2=-1)

    diagonal_covar = linear_operator.operators.DiagLinearOperator(var)

    dist_class = type(posterior.distribution)
    if dist_class is MultitaskMultivariateNormal:
        # Need to know if the original covairance was organised in an interleaved way so can instantiate correctly.
        # (see tests\sampling\helpers.py posterior_n3_t2_interleaved for example of interleaved variance).)
        if posterior._interleaved:  # noqa: SLF001
            return GPyTorchPosterior(MultitaskMultivariateNormal(mean, diagonal_covar, interleaved=True))

        return GPyTorchPosterior(MultitaskMultivariateNormal(mean, diagonal_covar, interleaved=False))

    return GPyTorchPosterior(dist_class(mean, diagonal_covar))
