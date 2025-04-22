# Currently mostly copied from TDR_rax/examples/dem2d/problem.py
# Update as simulator, BF QoI and make experiments are ready.
"""Helper file to put fundamental problem decisions in one place.

As users of the axtreme package we always need to:
1) Define the search space our problem lies within.
2) Define distribution we want to use to represent the output of the simulator.
3) Combine these with the simulator to create an Ax experiment.

Convenience items also defined here:
- importance datasets generated.

Guidance for creating/choosing the above is provided in `tutorials` (@melih)

Todo: TODO:
- (sw 2024_09_15): Once defined in this module, everything should be trated as a constant. Should all public things be
upper case? or is it enough that we all that stuff in problem is constant?
"""

# %%

import brute_force  # type: ignore[import]
import matplotlib.pyplot as plt
import numpy as np
import simulator  # type: ignore[import]
from ax import (
    Experiment,
    SearchSpace,
)
from ax.core import ParameterType, RangeParameter
from numpy.typing import NDArray
from scipy.stats import gumbel_r
from torch.utils.data import DataLoader, Dataset

from axtreme.data.dataset import MinimalDataset
from axtreme.experiment import make_experiment
from axtreme.simulator import utils as sim_utils
from axtreme.simulator.base import Simulator

# %%
### Pick the search space over which to create a surrogate
# TODO(@henrikstoklandberg): Decide on the search space.
# For now this is based on the min and max of the env data/long_term_distribution.npy
SEARCH_SPACE = SearchSpace(
    parameters=[
        RangeParameter(name="Hs", parameter_type=ParameterType.FLOAT, lower=7.5, upper=20),
        RangeParameter(name="Tp", parameter_type=ParameterType.FLOAT, lower=7.5, upper=20),
    ]
)

# %%
### Pick a distibution that you belive captures the noise behaviour of your simulator
DIST = gumbel_r

# %%
sim: Simulator = sim_utils.simulator_from_func(simulator.max_crest_height_simulator_function)

# Define the number of env samples that make a period
# _n_years_in_period = 10**4  # 10,000 years  # noqa: ERA001
_n_years_in_period = 100

_n_sea_states_in_year = 2922
_n_sea_states_in_period = _n_years_in_period * _n_sea_states_in_year
_sea_state_duration = 3 * 60 * 60  # 3 hours
_n_seconds_in_year = _n_sea_states_in_year * _sea_state_duration
_n_sea_states_in_period = _n_years_in_period * _n_seconds_in_year // _sea_state_duration

N_ENV_SAMPLES_PER_PERIOD = 1000  # Arbitrary number of env samples per period


# %%
### Automatically set up your experiment using the sim, search_space, and dist defined above.
def make_exp() -> Experiment:
    """Convience function returns a fresh Experiement of this problem."""
    # n_simulations_per_point can be changed, but it is typically a good idea to set it here so all QOIs and Acqusition
    # Functions are working on the same problem and are comparable
    return make_experiment(sim, SEARCH_SPACE, DIST, n_simulations_per_point=N_ENV_SAMPLES_PER_PERIOD)


# %%
# dataset and dataloader
dataset: Dataset[NDArray[np.float64]] = MinimalDataset(
    np.load(f"data/long_term_distribution_{_n_years_in_period}_years_no_hslim.npy")
)

dataloader = DataLoader(dataset, batch_size=256, shuffle=True)


# %%
# Get brute force QOI for this problem and period
chunck_brut_force_return_values, chunck_brut_force_return_mean, chunck_brut_force_return_variance = (
    brute_force.collect_or_calculate_results(
        dataloader,
        _n_sea_states_in_year,
        _n_sea_states_in_period,
        num_estimates=1_000,
    )
)
# %%
print(chunck_brut_force_return_values.shape)
# %%
# Plot brute force QOI
_ = plt.hist(chunck_brut_force_return_values, bins=100, density=True)
_ = plt.title("R-year return value distribution")  # type: ignore[assignment]
_ = plt.xlabel("R-year return value")  # type: ignore[assignment]
_ = plt.ylabel("Density")  # type: ignore[assignment]
plt.axvspan(
    chunck_brut_force_return_mean - chunck_brut_force_return_variance,
    chunck_brut_force_return_mean + chunck_brut_force_return_variance,
    alpha=0.5,
    color="red",
    label="variance",
)
_ = plt.axvline(chunck_brut_force_return_mean, color="red", label="mean")  # type: ignore[assignment]
_ = plt.legend()  # type: ignore[assignment]
plt.grid(True)  # noqa: FBT003
# %%
# TODO(@henrikstoklandberg): Add importance sampling dataset and dataloader
