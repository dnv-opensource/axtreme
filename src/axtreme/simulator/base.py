"""Define the simulator interface downsteam axtreme components expect."""

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Simulator(Protocol):
    """A protocol for (simulation) models."""

    def __call__(
        self, x: np.ndarray[tuple[int, int], np.dtype[np.float64]], n_simulations_per_point: int = 1
    ) -> np.ndarray[tuple[int, int, int], np.dtype[np.float64]]:
        """Evaluate the model at given points.

        Args:
            x: An array of shape (n_points, n_input_dims) of points at which to evaluate the model.
            n_simulations_per_point: The number of simulations to run at each point. Expected to have a default value.

        Returns:
            An array of shape (n_points, n_simulations_per_point, n_output_dims) of the model evaluated at the input
            points.
        """
        ...
