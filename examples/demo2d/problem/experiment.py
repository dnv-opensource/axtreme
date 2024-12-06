"""Contains the Ax Experiment object representing this problem.

Note: Generally the user needs to define this themselves from `env_data` and `simulator` objects, and decide on
the distribution and search spaces to use as part of that.
"""

# pyright: reportUnnecessaryTypeIgnoreComment=false

# %%
from ax import (
    Experiment,
    SearchSpace,
)
from ax.core import ParameterType, RangeParameter
from scipy.stats import gumbel_r

from axtreme.experiment import make_experiment as _make_experiment
from axtreme.simulator import Simulator
from axtreme.simulator import utils as sim_utils

# @ClaasRostock: I do this a bit thoughout the examples.<>.problem dirctories. Any tidier solutions?
# This allows us to run as interactive and as a module.
if __name__ == "__main__":
    import simulator  # type: ignore[import-not-found]
else:
    from . import simulator

# %%
### Pick the search space over which to create a surrogate
SEARCH_SPACE = SearchSpace(
    parameters=[
        RangeParameter(name="x1", parameter_type=ParameterType.FLOAT, lower=0, upper=1),
        RangeParameter(name="x2", parameter_type=ParameterType.FLOAT, lower=0, upper=1),
    ]
)

# %%
### Pick a distibution that you believe captures the noise behvaiour of your simulator
DIST = gumbel_r

### Define the simulator in the required interface
SIM: Simulator = sim_utils.simulator_from_func(simulator.dummy_simulator_function)


# ### Automatically set up you experiment using the sim, search_space, and dist defined above.
def make_experiment() -> Experiment:
    """Convience function return a fresh Experiement of this problem."""
    # TODO(sw 2024-11-19): set this to a lower value.
    return _make_experiment(SIM, SEARCH_SPACE, DIST, n_simulations_per_point=10_000)
