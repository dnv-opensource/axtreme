"""Obtain a brute force estimate of the Extreme Response Distribution (ERD).

This script creates brute force ERD samples for the importance sampling test case.
It uses the simulator and environment distribution defined in the sibling scripts.

The environment distribution is ``MultivariateNormal(mean=[0.1, 0.1], cov=0.2*I)``
truncated to positive values. The response follows a Gumbel distribution with loc and
scale determined by ``_true_underlying_func`` from ``simulator.py``.

Usage::

    python brute_force.py

Output:
    ``tests/qoi/data/importance_sampling/brute_force_solution.json``
"""

# pyright: reportUnnecessaryTypeIgnoreComment=false

# %%
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from numpy.typing import NDArray
from scipy.stats import gumbel_r
from torch.utils.data import DataLoader, RandomSampler, TensorDataset

torch.set_default_dtype(torch.float64)

# This allows us to run as interactive and as a module.
if __name__ == "__main__":
    from simulator import _true_underlying_func  # type: ignore[import-not-found]
else:
    from .simulator import _true_underlying_func
# for typing
_: Any

# %%
_results_dir: Path = Path(__file__).parent.parent
_results_file: Path = _results_dir / "brute_force_solution.json"


@dataclass
class ResultsObject:
    """The results object saved (as json) after brute force is run."""

    statistics: dict[str, float]
    samples: list[float]
    env_data: list[float]

    @classmethod
    def from_samples(cls, samples: torch.Tensor, env_data: torch.Tensor) -> "ResultsObject":
        """Create the object directly from samples."""
        statistics = {"median": float(samples.median()), "mean": float(samples.mean())}
        return ResultsObject(statistics=statistics, samples=samples.tolist(), env_data=env_data.tolist())


# %%
def collect_or_calculate_results(period_length: int, num_estimates: int = 2_000) -> tuple[torch.Tensor, torch.Tensor]:
    """Return saved results if available, otherwise calculate, save, and return them.

    Args:
        period_length: The number of environment samples per period of the ERD.
        num_estimates: The number of brute force estimates of the QoI. A new period is drawn for each estimate.

    Returns:
        Tuple of:
            ERD samples: (num_estimates,) samples of the ERD for that period length.
            X_max: (num_estimates, d) The environment location that produced the ERD sample.
    """
    samples = torch.tensor([])
    max_location = torch.tensor([])

    if _results_file.exists():
        with _results_file.open() as fp:
            results = json.load(fp)
            samples = torch.tensor(results["samples"])
            max_location = torch.tensor(results["env_data"])

    # Make additional samples if required
    if len(samples) < num_estimates:
        new_samples, new_max_location = brute_force(period_length, num_estimates - len(samples))

        samples = torch.concat([samples, new_samples])
        max_location = torch.concat([max_location, new_max_location])

        # Save results
        with _results_file.open("w") as fp:
            json.dump(asdict(ResultsObject.from_samples(samples, max_location)), fp)

    elif len(samples) > num_estimates:
        samples = samples[:num_estimates]
        max_location = max_location[:num_estimates]

    return samples, max_location


def brute_force(period_length: int, num_estimates: int = 2_000) -> tuple[torch.Tensor, torch.Tensor]:
    """Produce brute force samples of the Extreme Response Distribution.

    Args:
        period_length: The number of environment samples per period of the ERD.
        num_estimates: The number of brute force estimates of the QoI. A new period is drawn for each estimate.

    Returns:
        Tuple of:
            ERD samples: (num_estimates,) samples of the ERD for that period length.
            X_max: (num_estimates, d) The environment location that produced the ERD sample.
    """
    data: NDArray[np.float64] = np.load(_results_dir / "environment_distribution.npy")
    dataset = TensorDataset(torch.Tensor(data))

    dataloader = DataLoader(
        dataset,
        batch_size=4096,
        sampler=RandomSampler(dataset, num_samples=period_length, replacement=True),
    )

    return _brute_force_calc(dataloader, num_estimates)


def _brute_force_calc(
    dataloader: DataLoader[tuple[torch.Tensor, ...]], num_estimates: int = 2_000
) -> tuple[torch.Tensor, torch.Tensor]:
    """Calculate the QOI by brute force.

    Args:
        dataloader: The dataloader providing environment samples.
             - Each batch should have shape (batch_size, d)
             - The sum of the batch sizes returned by iterating through the dataloader should be a period length.
             - To get different results for each estimate, use e.g. a RandomSampler.
        num_estimates: The number of brute force estimates of the QoI.

    Returns:
        Tuple of:
            ERD samples: (num_estimates,) samples of the ERD for that period length.
            X_max: (num_estimates, d) The environment location that produced the ERD sample.
    """
    maxs = torch.zeros(num_estimates)

    _, d = next(iter(dataloader))[0].shape
    maxs_location = torch.zeros(num_estimates, d)

    for i in tqdm.tqdm(range(num_estimates)):
        current_max = float("-inf")

        for batch in dataloader:
            x = batch[0]
            params = _true_underlying_func(x)
            loc = params[:, 0].numpy()
            scale = params[:, 1].numpy()

            gumbel_samples: np.ndarray[tuple[int,], Any] = gumbel_r.rvs(loc=loc, scale=scale)  # type: ignore  # noqa: PGH003

            simulator_samples_max = gumbel_samples.max()
            if simulator_samples_max > current_max:
                current_max = simulator_samples_max

                max_index = np.argmax(gumbel_samples)
                maxs_location[i] = x[max_index, :]

        maxs[i] = current_max

    return maxs, maxs_location


# %%
if __name__ == "__main__":
    # %%
    N_ENV_SAMPLES_PER_PERIOD = 1000
    NUM_ESTIMATES = 100_000

    samples, x_max = collect_or_calculate_results(N_ENV_SAMPLES_PER_PERIOD, NUM_ESTIMATES)

    print(f"ERD samples shape: {samples.shape}")
    print(f"ERD median: {samples.median():.4f}")
    print(f"X_max shape: {x_max.shape}")

    # %%
    _ = plt.hist(samples, bins=100, density=True)
    _ = plt.axvline(samples.median(), color="red", linestyle="--", label=f"Median: {samples.median():.4f}")
    _ = plt.title(
        f"Extreme response distribution\n"
        f"(each result represents the largest value seen in a {N_ENV_SAMPLES_PER_PERIOD}-length period)"
    )
    _ = plt.xlabel("Response size")
    _ = plt.ylabel("Density")
    _ = plt.legend()
    plt.grid(True)  # noqa: FBT003
    plt.show()

    # %%
    _ = plt.scatter(x_max[:, 0], x_max[:, 1])
    _ = plt.title("Environment locations producing the maximum response")
    _ = plt.xlabel("x1")
    _ = plt.ylabel("x2")
    plt.show()

# %%
