"""Define the simulator for the importance sampling tutorial."""

# %%
from typing import cast

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.stats import gumbel_r
from torch.distributions import MultivariateNormal

from axtreme.simulator.base import Simulator

torch.set_default_dtype(torch.float64)


# %%
# These are helpers for our dummy simulator, and would not be available in a real problem
def _true_loc_func(x: NDArray[np.float64]) -> NDArray[np.float64]:
    # For this toy example we use a MultivariateNormal distribution
    loc = torch.tensor([1, 1])
    cov = torch.tensor([[0.03, 0], [0, 0.03]])

    dist = MultivariateNormal(loc=loc, covariance_matrix=cov)

    return np.exp(dist.log_prob(torch.tensor(x)).numpy())


def _true_scale_func(x: NDArray[np.float64]) -> NDArray[np.float64]:
    # For this toy example we use a constant scale for simplicity
    return np.ones(x.shape[0]) * 0.1


def dummy_simulator_function(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Generate a sample from a Gumbel distribution where the location and scale are function of X.

    Parameters:
        x: (n,2) array of points to simulate

    Returns:
        *(n,1) array of the simulator results for that point
    """
    location = _true_loc_func(x)
    scale = _true_scale_func(x)
    sample = cast("NDArray[np.float64]", gumbel_r.rvs(loc=location, scale=scale))
    return sample.reshape(-1, 1)


sim = dummy_simulator_function


class DummySimulatorSeeded(Simulator):
    """A seeded version of ``dummy_simulator_function`` conforming to the ``Simulator`` protocol.

    The each unique point in the x domain has a fixed seed used when generating samples. this can be
    useful for reproducibility. Points still appear "semi" random, as points close together use completely different
    seeds.

    Details:
        - Points which differ only after the 10th decimal place get the same random seed.
        - Co-ordinates at the same unique point will produce the exact same results. IT
    """

    def __call__(
        self, x: np.ndarray[tuple[int, int], np.dtype[np.float64]], n_simulations_per_point: int = 1
    ) -> np.ndarray[tuple[int, int, int], np.dtype[np.float64]]:
        """Evaluate the model at given points.

        Args:
            x: An array of shape (n_points, n_input_dims) of points at which to evaluate the model.
            n_simulations_per_point: The number of simulations to run at each point. Expected to have a default value
        Returns:
            An array of shape (n_points, n_simulations_per_point, n_output_dims) of the model evaluated at the input
            points.
        """
        # for each unque x point create a unqiue seed
        seeds = [DummySimulatorSeeded._hash_function(*tuple(x_i)) for x_i in x]
        location = _true_loc_func(x)
        scale = _true_scale_func(x)

        samples = []
        for loc_i, scale_i, seed_i in zip(location, scale, seeds, strict=True):
            sample = cast(
                "NDArray[np.float64]",
                gumbel_r.rvs(loc=loc_i, scale=scale_i, random_state=seed_i, size=n_simulations_per_point),
            )
            samples.append(sample)

        return np.expand_dims(np.stack(samples), axis=-1)

    @staticmethod
    def _hash_function(x1: float, x2: float) -> int:
        """Hash 2 float to a number within  between 0 and 2**32 - 1."""
        return abs(hash((x1, x2)) % (2**32 - 1))
