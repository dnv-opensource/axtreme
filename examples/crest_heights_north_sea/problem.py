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
from simulator import (  # type: ignore[import-not-found]
    MaxCrestHeightSimulatorSeeded,
    max_crest_height_simulator_function,
)
from torch.utils.data import DataLoader, Dataset
from usecase.env_data import collect_data  # type: ignore[import-not-found]

from axtreme.data import FixedRandomSampler, ImportanceAddedWrapper, MinimalDataset
from axtreme.experiment import make_experiment
from axtreme.qoi import MarginalCDFExtrapolation
from axtreme.sampling.ut_sampler import UTSampler
from axtreme.simulator.utils import simulator_from_func

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
sim = simulator_from_func(max_crest_height_simulator_function)

# %%
# TODO(hsb 2025-06-04): Should this be included here?I think it is nice to use the seeded simulator for reproducibility.
# Instantiate the seeded simulator
sim = MaxCrestHeightSimulatorSeeded()

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
# Number of simulations per point for each point added to the experiment.
# TODO(hsb 2025-06-04): Is the comment/explaination neccesary?
# Higher values will lead to less uncertainty in the GP fit, but will also increase the time it takes to run
# the experiment. Additionally, axtreme is meant to use few simulations per point, but high values can be useful for
# debugging and testing purposes.
N_SIMULATIONS_PER_POINT = 30


# %%
# Automatically set up your experiment using the sim, search_space, and dist defined above.
def make_exp() -> Experiment:
    """Convenience function returns a fresh Experiment of this problem."""
    # n_simulations_per_point can be changed, but it is typically a good idea to set it here so all QOIs and Acquisition
    # Functions are working on the same problem and are comparable
    return make_experiment(sim, SEARCH_SPACE, DIST, n_simulations_per_point=N_SIMULATIONS_PER_POINT)


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

# %% This is the result of `create_importance_samples.py` script.
importance_samples = torch.load("results/importance_sampling/importance_samples.pt", weights_only=True)
importance_weights = torch.load("results/importance_sampling/importance_weights.pt", weights_only=True)
importance_dataset = ImportanceAddedWrapper(MinimalDataset(importance_samples), MinimalDataset(importance_weights))
# %% This is based on the analysis performed in `qoi_bias_var.py` script.
sampler = FixedRandomSampler(
    importance_dataset,
    num_samples=8_000,
    seed=10,  # NOTE: we set a seed here for reproducibility, but this has not been cherry picked.
    replacement=True,
)
dataloader = DataLoader(importance_dataset, sampler=sampler, batch_size=256)

posterior_sampler = UTSampler()

# NOTE: While a constant, the input and output transforms still need to be attached for unique model the QoI runs on.
QOI_ESTIMATOR = MarginalCDFExtrapolation(
    env_iterable=dataloader,
    period_len=period_length,
    quantile=torch.exp(torch.tensor(-1)),
    quantile_accuracy=torch.tensor(0.01),
    posterior_sampler=posterior_sampler,
)
# %%
