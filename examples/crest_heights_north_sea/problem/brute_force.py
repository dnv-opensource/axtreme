"""Obtain a brute force estimate of the Extreme Response Distribution (ERD)."""

from typing import Any

import numpy as np
import tqdm
from simulator import max_crest_height_simulator_function
from torch import Tensor, zeros
from torch.utils.data import DataLoader


def brute_force_calc(dataloader: DataLoader[tuple[Tensor, ...]], num_estimates: int = 20) -> Tensor:
    """Calculate the QOI by brute force.

    Args:
        dataloader: The dataloader to use to get the environment samples.
             - Each batch should have shape (batch_size, d)
             - The sum of the batch sizes returned by iterating through the dataloader should be a period length
             - To get different results for each brute force estimate, the dataloader needs to give different
               data each time it is iterated through. This can be done by using e.g a RandomSampler.
        num_estimates: The number of brute force estimates of the QoI. A new period is drawn for each estimate.

    Returns:
        The QoI values calculated for each period. Shape (num_estimates,)
    """
    maxs = zeros(num_estimates)
    for i in tqdm.tqdm(range(num_estimates)):
        current_max = float("-inf")

        for batch in dataloader:
            samples = batch[0].to("cpu").numpy().reshape(1, 2)

            simulator_samples: np.ndarray[tuple[int,], Any] = max_crest_height_simulator_function(samples)  # type: ignore  # noqa: PGH003

            current_max = max(current_max, simulator_samples.max())

        maxs[i] = current_max

    return maxs
