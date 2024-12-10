"""Module for the MeanPosteriorSampler class."""

import torch
from botorch.posteriors import GPyTorchPosterior

from axtreme.sampling.base import PosteriorSampler


class MeanSampler(PosteriorSampler):
    """Class for sampling a posterior by returning the mean of the posterior.

    This is a simple sampler that just returns the mean of the posterior.
    """

    def __call__(self, posterior: GPyTorchPosterior) -> torch.Tensor:
        """Return the mean of the posterior."""
        # Unsqueezed first dimension so it conforms to the required dimension.
        # TODO(sw): does this conform for multi and single variate
        return posterior.mean.unsqueeze(0)
