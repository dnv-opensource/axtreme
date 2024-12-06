"""Helper file to put fundamental problem decisions in one place.

TODO(sw 2024-11-19): Extend this when solution to demo2d is implemented.
"""

# %%

import numpy as np
from numpy.typing import NDArray
from torch.utils.data import Dataset

from axtreme.data import MinimalDataset

from .problem import env_data

_data = env_data.collect_data()
dataset: Dataset[NDArray[np.float64]] = MinimalDataset(_data.to_numpy())
