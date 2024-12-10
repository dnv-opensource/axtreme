"""Define the simulator."""

from typing import cast

import numpy as np
import torch
from botorch.test_functions import BraninCurrin
from numpy.typing import NDArray
from scipy.stats import gumbel_r

_branin_currin = BraninCurrin(negate=False).to(dtype=torch.double)


# %%
# These are helpers for our dummy simulator, and would not be available in a real problme
def _true_loc_func(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return ((_branin_currin(torch.tensor(x)) / 20)[..., 0]).numpy()


def _true_scale_func(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return ((_branin_currin(1 - torch.tensor(x)) * 0.6)[..., 1]).numpy()


def dummy_simulator_function(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Generate a sample from a Gumbel distribution where the location and scale are function of X.

    Parameters:
        x: (n,2) array of points to simulate

    Returns:
        *(n,1) array of the simulator results for that point
    """
    location = _true_loc_func(x)
    scale = _true_scale_func(x)
    sample = cast(NDArray[np.float64], gumbel_r.rvs(loc=location, scale=scale))
    return sample.reshape(-1, 1)


sim = dummy_simulator_function
