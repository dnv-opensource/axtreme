"""Module for the NormalIndependentSampler class.

This module contains the NormalIndependentSampler class, which is a subclass of IndependentMCSampler.
It is used to randomly generate base samples from a Normal distribution and when you want to ignore
the covariance between different x points, and sample a single X output space independently.
"""

import torch
from botorch.posteriors import GPyTorchPosterior

from axtreme.sampling.independent_sampler import IndependentMCSampler


class NormalIndependentSampler(IndependentMCSampler):
    """IndependentMCSampler that randomly generates base samples from a Normal distribution."""

    # This static method could as well be a function in this module.
    # However, it is tightly coupled with the class structure and process, so it is placed here.
    @staticmethod
    def _required_base_sample_shape(
        sample_shape: torch.Size,
        posterior: GPyTorchPosterior,
    ) -> torch.Size:
        """Determine the required shape of base_samples.

        Args:
            sample_shape: The shape (e.g 'number' but in multidim) of samples desired.
            posterior: The posterior the base samples are applicable to

        Returns:
            torch.Size of (*sample_shape, t) where:
                - t is the number of targets for that posterior
        """
        explicit_shape = NormalIndependentSampler._explicit_posterior_shape(posterior)
        target_dimensionality = explicit_shape[-1]

        return torch.Size([*sample_shape, target_dimensionality])

    def _construct_base_samples(self, posterior: GPyTorchPosterior) -> None:
        """Build the base samples that should be used for sampling the GP output produced by a single input (x).

        `IndependentMCSampler`s apply the same base samples to every x in the posterior (independantly). The base
        samples to be used are define in this method.

        Args:
            posterior: the posterior to construct the base samples for.

        Return:
            None. Stores base_samples of shape (*b,n,t) in registered buffer.
        """
        required_base_sample_shape = self._required_base_sample_shape(self.sample_shape, posterior)
        if self.base_samples is None or self.base_samples.shape != required_base_sample_shape:
            # TODO(sw): Update this if future to use a generator?
            with torch.random.fork_rng():
                _ = torch.manual_seed(self.seed)
                base_samples = torch.randn(required_base_sample_shape)

            # Consforms to the nn.Module pattern,  means this will automatically be transfered to the right device.
            self.register_buffer("base_samples", base_samples)

        if self.base_samples.device != posterior.device:
            _ = self.to(device=posterior.device)  # pragma: nocover
        if self.base_samples.dtype != posterior.dtype:
            _ = self.to(dtype=posterior.dtype)
