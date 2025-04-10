# Currently mostly copied from TDR_rax/examples/dem2d/problem.py
# Update as simulator, BF QoI and make experiments are ready.
"""Helper file to put fundamental problem decisions in one place.

As users of the axtreme package we always need to:
1) Define the search space our problem lies within.
2) Define distribution we want to use to represent the output of the simulator.
3) Combine these with the simulator to create an Ax experiement.

Convience items also defined here:
- importance datasets generated.

Guidance for creating/choosing the above is provided in `tutorials` (@melih)

Todo: TODO:
- (sw 2024_09_15): Once defined in this module, everything should be trated as a constant. Should all public things be
upper case? or is it enough that we all that stuff in problem is constant?
"""

# %%

import numpy as np
from ax import (
    SearchSpace,
)
from ax.core import ParameterType, RangeParameter
from numpy.typing import NDArray
from scipy.stats import gumbel_r
from torch.utils.data import DataLoader, Dataset

from axtreme.data.dataset import MinimalDataset

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
### Pick a distibution that you belive captures the noise behvaiour of your simulator
DIST = gumbel_r

# %%
# dataset and dataloader
dataset: Dataset[NDArray[np.float64]] = MinimalDataset(np.load("data/long_term_distribution.npy"))

print("dataset.data: ", dataset.data)

dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

# %%
