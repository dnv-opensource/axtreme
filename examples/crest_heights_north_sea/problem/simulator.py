"""Define the simulator."""

from typing import cast

import numpy as np

# TODO (@am-kaiser): Had to remove the if main part to make problem.py work
# but we need to find a solution for the jupyter notebook example (AK 25-04-14)
from data.axtreme_case import (  # type: ignore[import-not-found]
    Tm01_from_Tp_gamma,
    Tm02_from_Tp_gamma,
    gamma_rpc205,
    omega_to_k_rpc205,
)
from data.wave_distributions import ForristallCrest  # type: ignore[import-not-found]

from axtreme.simulator.base import Simulator


def max_crest_height_simulator_function(
    x: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    water_depth: float = 110,
    sample_period: float = 3,
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
    km01 = omega_to_k_rpc205(2 * np.pi / tm01, water_depth)  # mean wave number corresponding to tm01

    num_waves_in_period = 3600 * sample_period / tm02

    forristall_crest = ForristallCrest(hs, tm01, km01, water_depth)

    # sample maximum crests in sample period
    c_max_in_period = forristall_crest.rvs_max(num_waves_in_period).ravel()

    return c_max_in_period.reshape(-1, 1)


# TODO(@am-kaiser): add seeded version of the class (AK 25-04-14)
class MaxCrestHeightSimulator(Simulator):
    """A class version of ``max_crest_height_gumbel_simulator_function`` conforming to the ``Simulator`` protocol.

    For each unique point in the x domain n_simulations_per_point samples are generated.
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
        samples = []
        for _ in np.arange(start=0, stop=n_simulations_per_point, step=1):
            sample = max_crest_height_simulator_function(x, water_depth, sample_period)
            samples.append(sample)

        result = np.stack(samples, axis=1)
        return cast("np.ndarray[tuple[int, int, int], np.dtype[np.float64]]", result)
