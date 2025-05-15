"""Define the simulator."""

# %%
from typing import cast

import numpy as np
from usecase.axtreme_case import (  # type: ignore[import-not-found]
    Tm01_from_Tp_gamma,
    Tm02_from_Tp_gamma,
    gamma_rpc205,
    omega_to_k_rpc205,
)
from usecase.wave_distributions import ForristallCrest  # type: ignore[import-not-found]

from axtreme.simulator.base import Simulator
from axtreme.simulator.utils import simulator_from_func

# water_depth and sample_period are fixed for the given problem
WATER_DEPTH = 110  # in meters
SAMPLE_PERIOD = 3  # in hours


def max_crest_height_simulator_function(
    x: np.ndarray[tuple[int, int], np.dtype[np.float64]],
) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
    """Generate the maximum crest height for an input environment x.

    Parameters:
        x: (n,2) array of points to simulate, corresponds to hs (significant wave height) and tp (peak wave period)
        water_depth: in meters
        sample_period: in hours

    Returns:
        *(n,1) array of the simulator results for that point
    """
    hs = x[:, 0]
    tp = x[:, 1]

    gamma = gamma_rpc205(hs, tp)
    tm01 = Tm01_from_Tp_gamma(tp, gamma)  # mean wave period
    tm02 = Tm02_from_Tp_gamma(tp, gamma)  # zero-crossing wave period
    km01 = omega_to_k_rpc205(2 * np.pi / tm01, WATER_DEPTH)  # mean wave number corresponding to tm01

    num_waves_in_period = 3600 * SAMPLE_PERIOD / tm02

    forristall_crest = ForristallCrest(hs, tm01, km01, WATER_DEPTH)

    # sample maximum crests in sample period
    c_max_in_period = forristall_crest.rvs_max(num_waves_in_period).ravel()

    return c_max_in_period.reshape(-1, 1)


# %%
class MaxCrestHeightSimulatorSeeded(Simulator):
    """A seeded version of the max_crest_height_simulator_function conforming to the ``Simulator`` protocol.

    Each unique point(before the 13th descimal) in the x domain has a fixed seed used when generating samples, which
    ensures reproducibility. Points still appear "semi" random, as points close together use completely different
    seeds. However running the simulator multiple times with the same input will produce the same results, but the
    results apear to be random like running the simulator without seeding.

    For a more detailed explanation of the seeding process, see the following issue #46:
    https://github.com/orgs/dnv-opensource/projects/4/views/1?pane=issue&itemId=108180911&issue=dnv-opensource%7Caxtreme%7C46
    Here plots are added to show the effect of this seeding process vs simply using a fixed seed.

    Details:
        - Points which differ only after the 13th decimal place get the same random seed.
        - Co-ordinates at the same unique point will produce the exact same results.
        - Points with even tiny differences produce completely different random sequences.
        - The overall effect is random-appearing but reproducible results for testing.
    """

    def __call__(
        self,
        x: np.ndarray[tuple[int, int], np.dtype[np.float64]],
        n_simulations_per_point: int = 1,
    ) -> np.ndarray[tuple[int, int, int], np.dtype[np.float64]]:
        """Evaluate the model at given points with reproducible results.

        Args:
            x: An array of shape (n_points, n_input_dims) of points at which to evaluate the model.
            n_simulations_per_point: The number of simulations to run at each point.
            water_depth: in meters
            sample_period: in hours
        Returns:
            An array of shape (n_points, n_simulations_per_point, n_output_dims) of the model evaluated at the input
            points.
        """
        # Initialize results array with correct shape
        result = np.zeros((x.shape[0], n_simulations_per_point, 1))

        # Create unique seed for each unique x point
        seeds = [MaxCrestHeightSimulatorSeeded._hash_function(*tuple(x_i)) for x_i in x]

        hs = x[:, 0]
        tp = x[:, 1]

        gamma = gamma_rpc205(hs, tp)
        tm01 = Tm01_from_Tp_gamma(tp, gamma)  # mean wave period
        tm02 = Tm02_from_Tp_gamma(tp, gamma)  # zero-crossing wave period
        km01 = omega_to_k_rpc205(2 * np.pi / tm01, WATER_DEPTH)  # mean wave number corresponding to tm01

        num_waves_in_period = 3600 * SAMPLE_PERIOD / tm02

        for i, (hs_i, tm01_i, km01_i, num_waves_i, seed_i) in enumerate(
            zip(hs, tm01, km01, num_waves_in_period, seeds, strict=True)
        ):
            # Create ForristallCrest with the specific parameters for this point
            forristall_crest = ForristallCrest(hs_i, tm01_i, km01_i, WATER_DEPTH)

            # Sample maximum crests in sample period with the seeded random state
            samples = forristall_crest.rvs_max(num_waves_i, size=n_simulations_per_point, seed=seed_i)

            result[i, :, 0] = samples

        return cast("np.ndarray[tuple[int, int, int], np.dtype[np.float64]]", result)

    @staticmethod
    def _hash_function(*args: float) -> int:
        """Hash float arguments to a number between 0 and 2**32 - 1."""
        return abs(hash(tuple(args)) % (2**32 - 1))


# %%
if __name__ == "__main__":
    # Quick and dirty tests:
    sim = simulator_from_func(max_crest_height_simulator_function)
    sim_seeded = MaxCrestHeightSimulatorSeeded()

    x = np.array([[2.0, 8.0], [3.0, 9.0]])

    # no seeding
    result1_no_seed = sim(x, n_simulations_per_point=5)
    result2_no_seed = sim(x, n_simulations_per_point=5)
    print("result1_no_seed", result1_no_seed[:2])
    print("result2_no_seed", result2_no_seed[:2])

    # The same value should produce the same result
    result1 = sim_seeded(x, n_simulations_per_point=5)
    result2 = sim_seeded(x, n_simulations_per_point=5)
    assert (result1 == result2).all(), "Same input should give same output"

    # Very similar values should produce different results
    x1 = np.array([[2.0 + 1e-5, 8.0], [3.0, 9.0]])
    result3 = sim_seeded(x1, n_simulations_per_point=5)
    # Should have different values due to different seeds
    print("result1", result1[0, 0, 0])
    print("result3", result3[0, 0, 0])
    assert not np.allclose(result1, result3, atol=1e-3), "Different inputs should give different outputs"

    print("Tests passed!")

    # %%
    # Plot the surface over a small area. If sample is not random the values should change slowly.
    # If simulator is seeded with unique seeds for each point in the x domain, the values should appear random,
    # but running the simulator multiple times should produce the same "random results".
    x1 = np.linspace(10.5, 10.5 + 1e-8, 10)
    x2 = np.linspace(10.5, 10.5 + 1e-8, 10)
    # Create a grid of (x, y) points
    x1_mesh, x2_mesh = np.meshgrid(x1, x2)
    x = np.column_stack([x1_mesh.flatten(), x2_mesh.flatten()])

    import matplotlib.pyplot as plt

    # Plot the surface using the simulator with no seed
    samples = sim(x).flatten()
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    _ = ax.scatter(x1_mesh, x2_mesh, samples.reshape(len(x1), len(x2)), cmap="viridis")

    samples_seeded1 = sim_seeded(x).flatten()
    samples_seeded2 = sim_seeded(x).flatten()
    assert (samples_seeded1 == samples_seeded2).all(), "Same input should give same output"

    # Plot the 2 different seeded results
    # The plots apear random, like in the simulator without seeding, but the results are the same
    # for the same input in the second run
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    _ = ax.scatter(x1_mesh, x2_mesh, samples_seeded1.reshape(len(x1), len(x2)), cmap="viridis")

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    _ = ax.scatter(x1_mesh, x2_mesh, samples_seeded2.reshape(len(x1), len(x2)), cmap="viridis")
