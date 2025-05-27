# Currently mostly copied from TDR_rax/examples/dem2d/problem.py
# Update as simulator, BF QoI and make experiments are ready.
"""Helper file to put fundamental problem decisions in one place.

As users of the axtreme package we always need to:
1) Define the search space our problem lies within.
2) Define distribution we want to use to represent the output of the simulator.
3) Combine these with the simulator to create an Ax experiment.

Todo: TODO:
- (sw 2024_09_15): Once defined in this module, everything should be treated as a constant. Should all public things be
upper case? or is it enough that we all that stuff in problem is constant?
"""

# %%

from pathlib import Path

import numpy as np
import torch
from ax import Experiment, ParameterConstraint, SearchSpace
from ax.core import ParameterType, RangeParameter
from brute_force import collect_or_calculate_results  # type: ignore[import-not-found]
from numpy.typing import NDArray
from scipy.stats import gumbel_r
from simulator import MaxCrestHeightSimulator  # type: ignore[import-not-found]
from torch.utils.data import Dataset
from usecase.env_data import collect_data

from axtreme.data.dataset import MinimalDataset
from axtreme.experiment import make_experiment

# %%
# Pick the search space over which to create a surrogate
hs_bounds = [0.1, 30]
tp_bounds = [1, 30]
SEARCH_SPACE = SearchSpace(
    parameters=[
        RangeParameter(name="Hs", parameter_type=ParameterType.FLOAT, lower=hs_bounds[0], upper=hs_bounds[1]),
        RangeParameter(name="Tp", parameter_type=ParameterType.FLOAT, lower=tp_bounds[0], upper=tp_bounds[1]),
    ],
    parameter_constraints=[
        # Linear constraint: Hs + Tp.lower_bound <= 1.5 Tp
        ParameterConstraint(constraint_dict={"Hs": 1, "Tp": -1.5}, bound=-tp_bounds[0]),
    ],
)

# %%
# Pick a distribution that you believe captures the noise behaviour of the simulator
DIST = gumbel_r

# %%
# Load simulator
sim = MaxCrestHeightSimulator()

# %%
# Load environment data
problem_dir = Path(__file__).resolve().parent
dataset: Dataset[NDArray[np.float64]] = MinimalDataset(collect_data().to_numpy())
# %%
# Convert usecase specific naming conventions to ax conventions
year_return_value = 10
n_sea_states_in_year = 2922

# In ax a period refers to a time length that a single extreme response relates to
# which is in this use case the number of sea states in the desired return period
period_length = year_return_value * n_sea_states_in_year


# %%
# Automatically set up your experiment using the sim, search_space, and dist defined above.
def make_exp() -> Experiment:
    """Convenience function returns a fresh Experiment of this problem."""
    # n_simulations_per_point can be changed, but it is typically a good idea to set it here so all QOIs and Acquisition
    # Functions are working on the same problem and are comparable
    return make_experiment(sim, SEARCH_SPACE, DIST, n_simulations_per_point=30)


# %%
# Get brute force QOI for this problem and period
extreme_response_values, _ = collect_or_calculate_results(
    period_length,
    num_estimates=100_000,  # each estimate draws a period_length samples
)

# Exp(-1) quantile of the ERD is used to convert to the "return value"
# For example: the exp(-1) quantile of the 20 year period ERD give the 20 year return value.
# This is based on discussion with Odin, and the following paper: https://www.duo.uio.no/bitstream/handle/10852/101693/JOMAE2022_TSsim_rev1.pdf?sequence=1
brute_force_qoi = torch.quantile(extreme_response_values, np.exp(-1))

# %%
# TODO(@henrikstoklandberg): Add importance sampling dataset and dataloader
