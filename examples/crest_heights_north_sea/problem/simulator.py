"""Define the simulator."""

import numpy as np

from axtreme.simulator.base import Simulator

from .data.axtreme_case import Tm01_from_Tp_gamma, Tm02_from_Tp_gamma, gamma_rpc205, omega_to_k_rpc205
from .data.wave_distributions import ForristallCrest


def max_crest_height_simulator_function(
    x: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    water_depth: float = 110,
    sample_period: float = 3,
) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
    """Generate a sample from a Gumbel distribution where the location and scale are a function of x.

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
    km01 = omega_to_k_rpc205(2 * np.pi / tm01, water_depth)  # mean wave number corresponding to tm01

    num_waves_in_period = 3600 * sample_period / tm02

    forristall_crest = ForristallCrest(hs, tm01, km01, water_depth)

    # sample maximum crests in sample period
    c_max_in_period = forristall_crest.rvs_max(num_waves_in_period).ravel()

    return c_max_in_period.reshape(-1, 1)


class MaxCrestHeightSimulator(Simulator):
    """A seeded version of ``max_crest_height_gumbel_simulator_function`` conforming to the ``Simulator`` protocol.

    The each unique point in the x domain has a fixed seed used when generating samples. this can be
    useful for reproducibility. Points still appear "semi" random, as points close together use completly different
    seeds.

    Details:
        - Points which differ only after the 10th decimal place get the same random seed.
        - Co-ordinates at the same unique point will produce the exact same results. IT
    """

    def __call__(
        self,
        x: np.ndarray[tuple[int, int], np.dtype[np.float64]],
        n_simulations_per_point: int = 1,
        water_depth: float = 110,
        sample_period: float = 3,
    ) -> np.ndarray[tuple[int, int, int], np.dtype[np.float64]]:
        """Evaluate the model at given points.

        Args:
            x: An array of shape (n_points, n_input_dims) of points at which to evaluate the model.
            n_simulations_per_point: The number of simulations to run at each point. Expected to have a default value
            water_depth: in meters
            sample_period: in hours
        Returns:
            An array of shape (n_points, n_simulations_per_point, n_output_dims) of the model evaluated at the input
            points.
        """
        # for each unique x point create a unique seed
        seeds = np.linspace(0, n_simulations_per_point, 1, dtype=int)  # AK: does not do anything so far

        samples = []
        for seed in seeds:
            _ = np.random.default_rng(seed)
            sample = max_crest_height_simulator_function(x, water_depth, sample_period)
            samples.append(sample)

        return np.expand_dims(np.stack(samples), axis=-1)  # AK ToDo: need too check if the dimensions are correct
