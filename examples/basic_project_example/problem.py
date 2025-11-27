"""Helper file to put fundamental problem decisions for the example use case in one place.

This file includes the following:
1) Definition of the search space for which the surrogate model is created
2) Definition of the distribution that best captures the simulators response
3) Loading  simulator
4) Load environment data
5) Calculation of brute force solution
6) Calculation of QoI

Future projects can use this structure as a guideline on how an axtreme use case can be set up.

Variables and functions from this file are used in the DOE calculations (doe.py).

"""

# %%
from pathlib import Path

import numpy as np
import torch
from ax import Experiment, SearchSpace
from ax.core import ParameterType, RangeParameter
from brute_force import collect_or_calculate_results  # type: ignore[import-not-found]
from env_data import collect_data  # type: ignore[import-not-found]
from scipy.stats import gumbel_r
from simulator import DummySimulatorSeeded
from torch.utils.data import DataLoader

from axtreme.data import FixedRandomSampler, MinimalDataset
from axtreme.experiment import make_experiment
from axtreme.qoi import MarginalCDFExtrapolation
from axtreme.sampling.ut_sampler import UTSampler

torch.set_default_dtype(torch.float64)

# %%
# Pick the search space over which to create a surrogate
SEARCH_SPACE = SearchSpace(
    parameters=[
        RangeParameter(name="x1", parameter_type=ParameterType.FLOAT, lower=0, upper=1),
        RangeParameter(name="x2", parameter_type=ParameterType.FLOAT, lower=0, upper=1),
    ]
)

# %%
# Pick a distribution that you believe captures the noise behaviour of the simulator
DIST = gumbel_r

# %%
# For now the seeded simulator is used for reproducibility of the results over different runs.
# This is useful for development and debugging purposes, but in production you might want to use the non-seeded version.
# This is because the seeded simulator will always return the same results for the same input parameters.
# If you want to use the non-seeded simulator, use:
# sim = simulator_from_func(max_crest_height_simulator_function)  # noqa: ERA001

# Instantiate the seeded simulator
SIM = DummySimulatorSeeded()

# %%
# Load environment data
problem_dir = Path(__file__).resolve().parent
dataset = MinimalDataset(collect_data().to_numpy())

# %%
# Convert usecase specific naming conventions to ax conventions
year_return_value = 10
n_states_in_year = 1000

# In ax a period refers to a time length that a single extreme response relates to
# which is in this use case the number of states in the desired return period
PERIOD_LENGTH = year_return_value * n_states_in_year

# %%
# Number of simulations per point for each point added to the experiment.
# Higher values will lead to less uncertainty in the GP fit, but will also increase the time it takes to run
# the experiment. Additionally, axtreme is meant to use in order to reduce the need to run
# computationally expensive simulators, so few simulations per point is preferred in real scenarios.
# However high number of simulations per point can be useful for debugging and testing purposes, as it will efectivly
# reduce the uncertainty in the GP, at the cost of increased runtime.
N_SIMULATIONS_PER_POINT = 30


# %%
# Automatically set up your experiment using the sim, search_space, and dist defined above.
def make_exp() -> Experiment:
    """Convenience function that returns a fresh Experiment of this problem."""
    # n_simulations_per_point can be changed, but it is typically a good idea to set it here so all QOIs and Acquisition
    # Functions are working on the same problem and are comparable
    return make_experiment(SIM, SEARCH_SPACE, DIST, n_simulations_per_point=N_SIMULATIONS_PER_POINT)


# %%
# Get brute force QOI for this problem and period
extreme_response_values, _ = collect_or_calculate_results(
    PERIOD_LENGTH,
    num_estimates=100_000,  # each estimate draws a period_length samples
)

# Exp(-1) quantile of the ERD is used to convert to the "return value"
# For example: the exp(-1) quantile of the 20 year period ERD give the 20 year return value.
# This is based on discussion with Odin, and the following paper: https://www.duo.uio.no/bitstream/handle/10852/101693/JOMAE2022_TSsim_rev1.pdf?sequence=1
brute_force_qoi = torch.quantile(extreme_response_values, np.exp(-1))

# %%
sampler = FixedRandomSampler(
    dataset,
    num_samples=8_000,  # Value chosen based on bias-variance analysis.
    seed=10,  # NOTE: we set a seed here for reproducibility, but this has not been cherry picked.
    replacement=True,
)
dataloader = DataLoader(dataset, sampler=sampler, batch_size=256)

posterior_sampler = UTSampler()

# NOTE: While a constant, the input and output transforms still need to be attached for unique model the QoI runs on.
QOI_ESTIMATOR = MarginalCDFExtrapolation(
    env_iterable=dataloader,
    period_len=PERIOD_LENGTH,
    quantile=torch.exp(torch.tensor(-1)),
    quantile_accuracy=torch.tensor(0.01),
    posterior_sampler=posterior_sampler,
)
# %%
