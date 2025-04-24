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
import simulator  # type: ignore[import]
from ax import (
    Experiment,
    SearchSpace,
)
from ax.core import ParameterType, RangeParameter
from scipy.stats import gumbel_r

from axtreme.experiment import make_experiment
from axtreme.simulator import utils as sim_utils
from axtreme.simulator.base import Simulator

# %%
# Pick the search space over which to create a surrogate
# TODO(@henrikstoklandberg): Decide on the search space.
# For now this is based on the min and max of the env data/long_term_distribution.npy
SEARCH_SPACE = SearchSpace(
    parameters=[
        RangeParameter(name="Hs", parameter_type=ParameterType.FLOAT, lower=7.5, upper=20),
        RangeParameter(name="Tp", parameter_type=ParameterType.FLOAT, lower=7.5, upper=20),
    ]
)

# %%
# Pick a distribution that you believe captures the noise behaviour of the simulator
DIST = gumbel_r

# %%
# Load simulator
sim: Simulator = sim_utils.simulator_from_func(simulator.max_crest_height_simulator_function)

# %%
# Convert usecase specific naming conventions to ax conventions
year_return_value = 10
n_sea_states_in_year = 2922

# In ax a period refers to a time length that a single extreme response relates to
# which is in this use case the number of sea states in the desired return period
period_length = year_return_value * n_sea_states_in_year

# %%
# Set axtreme specific parameters
N_ENV_SAMPLES_PER_PERIOD = 1000  # The number of simulations to run for each point in the experiment.
num_estimates = 20  # The number of brute force estimates of the QoI. A new period is drawn for each estimate.


# %%
# Automatically set up your experiment using the sim, search_space, and dist defined above.
def make_exp() -> Experiment:
    """Convience function returns a fresh Experiement of this problem."""
    # n_simulations_per_point can be changed, but it is typically a good idea to set it here so all QOIs and Acqusition
    # Functions are working on the same problem and are comparable
    return make_experiment(sim, SEARCH_SPACE, DIST, n_simulations_per_point=N_ENV_SAMPLES_PER_PERIOD)


# %%
# Get brute force QOI for this problem and period
extrem_response_values, extrem_response_mean, extrem_response_variance = brute_force.collect_or_calculate_results(
    period_length,
    num_estimates=num_estimates,
)


# %%
# TODO(@henrikstoklandberg): Add importance sampling dataset and dataloader
