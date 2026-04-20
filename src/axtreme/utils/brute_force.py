from collections.abc import Callable

import torch
import tqdm
from torch.distributions import Distribution
from torch.utils.data import DataLoader


def brute_force_calc(
    dataloader: DataLoader[tuple[torch.Tensor, ...]],
    response_params_func: Callable[[torch.Tensor], torch.Tensor],
    response_dist_class: type[Distribution],
    num_estimates: int = 2_000,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Calculate the QOI by brute force.

    Args:
        dataloader: Provides environment samples of shape (batch_size, d), where batches concatenate to shape
            (period_length, d). The dataloader will be iterated through num_estimates times, and needs to give different
            data each time it is iterated through (e.g., by using a RandomSampler).
            ```python
            dataloader = DataLoader(
                dataset,
                batch_size=4096,
                sampler=RandomSampler(dataset, num_samples=period_length, replacement=True),
            )
            ```
        response_params_func: Take env samples (batch_size, d) and return parameters of the response distribution
            (batch_size, num_params).
        response_dist_class: The distribution class to use to get response samples. Expect to have single output.
        num_estimates: The number of brute force estimates of the QoI. A new period is drawn for each estimate.

    Returns:
        Tuple of:
            ERD samples: (num_estimates,) samples of the ERD for that period length. QoIs can be calculated from this.
            X_max: (num_estimates, d) The location in the environments space that produced the ERD sample.

    Note:
        - This produces the same results as the methods used in `examples`. Examples should eventually be updated to use
        this function. Further testing should probably be completed because this is used to get "true values".
    """
    maxs = torch.zeros(num_estimates)

    _, d = next(iter(dataloader)).shape
    maxs_location = torch.zeros(num_estimates, d)

    for i in tqdm.tqdm(range(num_estimates)):
        current_max = float("-inf")

        for batch in dataloader:
            assert batch.ndim() == 2, f"Expected batch shape (batch_size, {d}), got {batch.shape}"  # noqa: PLR2004
            params = response_params_func(batch)
            response_dist = response_dist_class(*torch.unbind(params, dim=-1))

            # Should produce shape (batch_size,)
            samples = response_dist.sample()

            # dim=0 is added so get both value and indices
            simulator_samples_max = samples.max(dim=0)
            if simulator_samples_max.values > current_max:
                current_max = simulator_samples_max.values
                maxs_location[i] = batch[simulator_samples_max.indices, :]

        maxs[i] = current_max

    return maxs, maxs_location
