"""Simulator for the importance sampling test data.

This simulator wraps the true underlying function used in
``test_marginal_cdf_extrapolation.test_system_marginal_cdf_with_importance_sampling``.

The underlying function defines a Gumbel response distribution where:
- loc(x) = exp(log_prob(x)) with x ~ MultivariateNormal(mean=[1, 1], cov=diag(0.03))
- scale(x) = 0.1 (constant)

The response at each environment point is a sample from Gumbel(loc(x), scale(x)).
"""

# %%
from typing import cast

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.stats import gumbel_r
from torch.distributions import MultivariateNormal

from axtreme.simulator.base import Simulator

torch.set_default_dtype(torch.float64)


def _true_underlying_func(x: torch.Tensor) -> torch.Tensor:
    """Compute the Gumbel distribution parameters (loc, scale) for each input point.

    This is the same function used in the test as the deterministic GP mean function.

    Args:
        x: Input tensor of shape (n_points, 2).

    Returns:
        Tensor of shape (n_points, 2) where columns are [loc, scale].
    """
    dist_mean, dist_cov = torch.tensor([1, 1]), torch.tensor([[0.03, 0], [0, 0.03]])
    dist = MultivariateNormal(loc=dist_mean, covariance_matrix=dist_cov)
    loc = torch.exp(dist.log_prob(x))

    scale = torch.ones(x.shape[0]) * 0.1

    return torch.stack([loc, scale], dim=-1)


class ImportanceSamplingTestSimulator(Simulator):
    """A seeded simulator for the importance sampling test.

    At each environment point x, the response follows a Gumbel distribution with location and scale
    determined by ``_true_underlying_func``. Each unique point gets a deterministic seed for reproducibility.
    """

    def __call__(
        self, x: np.ndarray[tuple[int, int], np.dtype[np.float64]], n_simulations_per_point: int = 1
    ) -> np.ndarray[tuple[int, int, int], np.dtype[np.float64]]:
        """Evaluate the simulator at given points.

        Args:
            x: An array of shape (n_points, n_input_dims) of points at which to evaluate the model.
            n_simulations_per_point: The number of simulations to run at each point.

        Returns:
            An array of shape (n_points, n_simulations_per_point, n_output_dims) of the model evaluated at the input
            points.
        """
        params = _true_underlying_func(torch.tensor(x))
        loc = params[:, 0].numpy()
        scale = params[:, 1].numpy()

        seeds = [self._hash_function(*tuple(x_i)) for x_i in x]

        samples = []
        for loc_i, scale_i, seed_i in zip(loc, scale, seeds, strict=True):
            sample = cast(
                "NDArray[np.float64]",
                gumbel_r.rvs(loc=loc_i, scale=scale_i, random_state=seed_i, size=n_simulations_per_point),
            )
            samples.append(sample)

        return np.expand_dims(np.stack(samples), axis=-1)

    @staticmethod
    def _hash_function(x1: float, x2: float) -> int:
        """Hash 2 floats to an integer between 0 and 2**32 - 1."""
        return abs(hash((x1, x2)) % (2**32 - 1))


# %%
if __name__ == "__main__":
    # %%
    sim = ImportanceSamplingTestSimulator()
    x = np.array([[1.0, 1.0], [0.5, 0.5], [0.0, 0.0]])
    result = sim(x, n_simulations_per_point=3)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {result.shape}")
    print(f"Results:\n{result}")

    # Verify reproducibility
    assert (sim(x, n_simulations_per_point=5) == sim(x, n_simulations_per_point=5)).all()
    print("Reproducibility check passed.")

# %%
