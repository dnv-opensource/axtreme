"""Define the simulator."""

# %%
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy.typing import NDArray
from scipy.stats import gumbel_r
from torch.distributions import Categorical, MixtureSameFamily, MultivariateNormal

from axtreme.simulator.base import Simulator

torch.set_default_dtype(torch.float64)


# %%
# These are helpers for our dummy simulator, and would not be available in a real problem
def _true_loc_func(x: NDArray[np.float64]) -> NDArray[np.float64]:
    # For this toy example we use a Mixture distribution of a MultivariateNormal distribution
    dist1_mean, dist1_cov = torch.tensor([0.8, 0.8]), torch.tensor([[0.03, 0], [0, 0.03]])
    dist2_mean, dist2_cov = torch.tensor([0.2, 0.8]), torch.tensor([[0.04, 0.01], [0.01, 0.04]])
    dist3_mean, dist3_cov = torch.tensor([0.5, 0.2]), torch.tensor([[0.06, 0], [0, 0.06]])

    locs = torch.stack([dist1_mean, dist2_mean, dist3_mean])
    covs = torch.stack([dist1_cov, dist2_cov, dist3_cov])
    component_dist = MultivariateNormal(loc=locs, covariance_matrix=covs)

    mix = Categorical(
        torch.ones(
            3,
        )
    )
    gmm = MixtureSameFamily(mix, component_dist)
    return np.exp(gmm.log_prob(torch.tensor(x)).numpy())


def _true_scale_func(x: NDArray[np.float64]) -> NDArray[np.float64]:
    # For this toy example we use a constant scale for simplicity
    return np.ones(x.shape[0]) * 0.1


def dummy_simulator_function(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Generate a sample from a Gumble distribution where the location and scale are function of X.

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
        # for each unique x point create a unique seed
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


# %%
if __name__ == "__main__":
    # Quick and dirty tests:
    sim = DummySimulatorSeeded()
    x = np.array([[0.5000, 0.5], [0.3, 0.3]])
    # The same value will produce the same result
    assert (sim(x, n_simulations_per_point=5) == sim(x, n_simulations_per_point=5)).all()

    # %%
    # Very similar values produce different results
    # we allow a wide margin of error because results should be completely different due to sampling
    x1 = np.array([[0.5 + 1e-5, 0.5], [0.3 + 1e-5, 0.3]])
    assert (sim(x1, n_simulations_per_point=5) != sim(x, n_simulations_per_point=5)).all()

    # %%
    # Plot the surface over a small area. If sample is not random the values should change slowly.
    x1 = np.linspace(0.5, 0.5 + 1e-8, 100)  # 100 points between -5 and 5
    x2 = np.linspace(0.5, 0.5 + 1e-8, 100)
    # Create a grid of (x, y) points
    x1_mesh, x2_mesh = np.meshgrid(x1, x2)
    x = np.column_stack([x1_mesh.flatten(), x2_mesh.flatten()])

    samples = sim(x).flatten()
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    _ = ax.scatter(x1_mesh, x2_mesh, samples.reshape(len(x1), len(x2)))

# %%
