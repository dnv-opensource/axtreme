"""Module for the UTSampler class for sampling a posterior using the Unscented Transform."""

import numpy as np
import torch
from botorch.posteriors import Posterior
from filterpy.kalman import MerweScaledSigmaPoints

from axtreme.sampling.independent_sampler import IndependentMCSampler


class UTSampler(IndependentMCSampler):
    """Class for sampling a posterior using the Unscented Transform.

    The Unscented Transform is a method for transforming a distribution through a non-linear function.
    The method is based on the Unscented Kalman Filter.
    It uses a set of sigma points to estimate the mean and covariance of the transformed distribution.
    """

    def __init__(
        self,
        alpha: float = 1,
        beta: float = 2,
        kappa_base: float = 3,
    ) -> None:
        """Initializer for the UTSampler.

        Args:
            alpha: Scaling parameter for the sigma points. See filterpy.kalman.MerweScaledSigmaPoints.
            beta: Parameter for the distribution.
                - For Gaussian distributions, beta=2 is optimal.
                - See filterpy.kalman.MerweScaledSigmaPoints.
            kappa_base: Base parameter for the sigma points. See filterpy.kalman.MerweScaledSigmaPoints.
        """
        # We don't want to use the .__init__() on MCSampler as to forces `sample_shape` which is currently unknown.
        # Instead we use nn.Module (root class) because it performs internal set up.
        torch.nn.Module.__init__(self)

        # This is part of the design contact specific by MCSampler. We want to conform but it irrelevant for this class.
        self.seed = None

        # This is part of the design contact specific by MCSampler. We want to conform but need to create it later
        self.sample_shape: torch.Size | None = None

        # Make these private. Users should create a new sampler if they want different parameters
        self._alpha = alpha
        self._beta = beta
        self._kappa_base = kappa_base

        self.register_buffer("base_samples", None)

    @property
    def alpha(self) -> float:
        """The alpha parameter defined by filterpy.kalman.MerweScaledSigmaPoints."""
        return self._alpha

    @property
    def beta(self) -> float:
        """The beta parameter defined by filterpy.kalman.MerweScaledSigmaPoints."""
        return self._beta

    @property
    def kappa_base(self) -> float:
        """The kappa_base parameter related to kappa defined by filterpy.kalman.MerweScaledSigmaPoints.

        The kappa value used in filterpy.kalman.MerweScaledSigmaPoints is calculated as `kappa_base - num_target`.
        """
        return self._kappa_base

    def _construct_base_samples(self, posterior: Posterior) -> None:
        """Build the base samples that should be used for sampling the GP output produced by a single input (x).

        `IndependentMCSampler`s apply the same base samples to every x in the posterior (independantly). The base
        samples to be used are define in this method.

        Args:
            posterior: the posterior to construct the base samples for.

        Return:
            None. Stores base_samples of shape (*b,n,m) in registered buffer.
        """
        # has shape (*b,n,m)
        explicit_posterior_shape = self._explicit_posterior_shape(posterior)
        # This is the t dim
        target_dim = explicit_posterior_shape[-1]

        if (self.base_samples is None) or (self.base_samples.shape[-1] != target_dim):
            points, mean_weights, cov_weights = calculate_sigmas(target_dim, self.alpha, self.beta, self.kappa_base)

            self.sample_shape = torch.Size([points.shape[0]])

            self.register_buffer("base_samples", points)
            self.register_buffer("mean_weights", mean_weights)
            self.register_buffer("cov_weights", cov_weights)

            for attr_name in ["base_samples", "mean_weights", "cov_weights"]:
                attr = getattr(self, attr_name)
                if attr.device != posterior.device:
                    _ = self.to(device=posterior.device)
                if attr.dtype != posterior.dtype:
                    _ = self.to(dtype=posterior.dtype)

    def mean(self, transformed_points: torch.Tensor, dim: int = 0) -> torch.Tensor:
        """Estimate the mean of the transformed UT sampled points.

        The UT sampled points are designed together with the weights to estimate the mean of the distribution
        When the points are transformed through some function,
        the mean of the transformed points is estimated by calculating the weighted sum of the points.

        This will only work correctly if the last points generated using this sampler
         - Have the same number of targets (m) the transformed points.
         - Were generated using the same alpha, beta, and kappa_base parameters.

        Args:
            transformed_points: The transformed points to estimate the mean of.
            dim: The dimension along which to calculate the mean.

        Returns:
            The estimated mean of the distribution.
        """
        if transformed_points.shape[dim] != self.mean_weights.shape[0]:
            msg = f"Expected {self.mean_weights.shape[0]} points, got {transformed_points.shape[0]}"
            raise ValueError(msg)

        return torch.tensordot(self.mean_weights[None], transformed_points.moveaxis(dim, 0), dims=1).squeeze(0)

    def var(self, transformed_points: torch.Tensor, dim: int = 0) -> torch.Tensor:
        """Estimate the variance of the transformed UT sampled points.

        The UT sampled points are designed together with the weights to estimate the variance of the distribution
        When the points are transformed through some function,
        the variance of the transformed points is estimated by calculating the weighted sum of the points.

        This will only work correctly if the last points generated using this sampler
         - Have the same number of targets (m) as the transformed points.
         - Were generated using the same alpha, beta, and kappa_base parameters.

        Args:
            transformed_points: The transformed points to estimate the variance of.
            dim: The dimension along which to calculate the variance.

        Returns:
            The estimated variance of the distribution.
        """
        moved_axis_points = transformed_points.moveaxis(dim, 0)

        if moved_axis_points.shape[0] != self.mean_weights.shape[0]:
            msg = f"Expected {self.mean_weights.shape[0]} points, got {moved_axis_points.shape[0]}"
            raise ValueError(msg)

        if moved_axis_points.shape[0] != self.cov_weights.shape[0]:
            msg = f"Expected {self.cov_weights.shape[0]} points, got {moved_axis_points.shape[0]}"
            raise ValueError(msg)

        centered_points = moved_axis_points - self.mean(moved_axis_points, dim=0)[None]
        return torch.tensordot(self.cov_weights[None], torch.square(centered_points), dims=1).squeeze(0)


# This function is not part of the class because it is not specific to the class.
# However, it could be moved to the class if that is deemed more appropriate.
def calculate_sigmas(
    dim: int, alpha: float, beta: float, kappa_base: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create the sigma points, and the weight for calculating mean and variance.

    Args:
        dim: dimensionality of the space to produce sigma points for.
            E.g The dimensionality of the output of a GP at a sinlge input point.
        alpha: Scaling parameter for the sigma points.
        beta: Parameter for the distribution. For Gaussian distributions, beta=2 is optimal.
        kappa_base: Base parameter for the sigma points.

    Return:
        A tuple of three tensors where,

        1. (m,dim) defines signma points, where
           - m: is the number of sigma points generated
           - dim: is the dimension of those points
        2. (m): weights required to combines these signma points for a mean estimate
        3. (m): weights required to combines these signma points for a variance estimate

    Todo:
        - does not currently support correlation between t dimensions.
        - (ks) - For now using filterpy for UT. Might want to implement our own version
    """
    sigmas = MerweScaledSigmaPoints(n=dim, alpha=alpha, beta=beta, kappa=kappa_base - dim)

    points = torch.tensor(sigmas.sigma_points(np.zeros(dim), np.eye(dim)))
    mean_weights = torch.tensor(sigmas.Wm)
    cov_weights = torch.tensor(sigmas.Wc)

    return points, mean_weights, cov_weights
