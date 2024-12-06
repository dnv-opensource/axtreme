"""Module for quantity of interest Quantity of Interest estimator protocol."""

from typing import Protocol

import torch
from botorch.models.model import Model


class QoIEstimator(Protocol):
    """A protocol for quantity of interest (QoI) estimators."""

    def __call__(self, model: Model) -> torch.Tensor:
        """Function that produces multiple estimates of a scalar quantity of interest.

        If the function needs additional arguments,
        this should be handled by a class that implements this protocol or a closure.

        Args:
            model: A botorch model that is used to estimate the QoI.

        Returns:
            torch.Tensor: A tensor of QoI estimates with shape (number_of_estimates,).
        """
        ...

    def mean(self, x: torch.Tensor) -> torch.Tensor:
        """Function that computes the mean of the QoI estimates (the output of the call() method).

        For many applications this should just be using a default implementation that computes the mean.
        E.g using torch.mean(x).

        However, in some special cases, it might be useful to provide a custom implementation
        to give a more accurate estimate. E.g. when UTSampler is used.

        Args:
            x: A tensor of QoI estimates with shape (number_of_estimates,).

        Returns:
            torch.Tensor: The mean of the QoI estimates. Should be a scalar.
        """
        return torch.mean(x)

    def var(self, x: torch.Tensor) -> torch.Tensor:
        """Function that computes the variance of the QoI estimates (the output of the `call()` method).

        For many applications this should just be using a default implementation that computes the variance.
        E.g. using `torch.var(x)`.

        However, in some special cases, it might be useful to provide a custom implementation
        to give a more accurate estimate. E.g. when UTSampler is used.

        Args:
            x: A tensor of QoI estimates with shape (number_of_estimates,).

        Returns:
            torch.Tensor: The variance of the QoI estimates. Should be a scalar.
        """
        return torch.var(x)
