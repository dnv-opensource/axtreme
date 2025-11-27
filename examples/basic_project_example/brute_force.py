"""Obtain a brute force estimate of the Extreme Response Distribution (ERD).

This helper file creates samples of the ERD for a given period length (by running this file). It saves those results so
they can be immediately retrieved when users want to explore this use case. It also provides the helper
`collect_or_calculate_results` which will collect pre-existing results if they exist, and will otherwise calculate,
save and return results.

Brute force values are available for this use case because we create a mock simulator. For real world problems this is
typically not possible.
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
from simulator import _true_loc_func, _true_scale_func
from torch.utils.data import DataLoader, RandomSampler, TensorDataset

torch.set_default_dtype(torch.float64)

_: Any


# %%
_results_dir: Path = Path(__file__).parent / "results" / "brute_force"
if not _results_dir.exists():
    _results_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class ResultsObject:
    "The results object saved (as json) after brute force is run."

    # statistics are optional
    statistics: dict[str, float]
    samples: list[float]
    env_data: list[float]

    @classmethod
    def from_samples(cls, samples: torch.Tensor, env_data: torch.Tensor) -> "ResultsObject":
        """Create the object directly from samples."""
        statistics = {"median": float(samples.median()), "mean": float(samples.mean())}
        return ResultsObject(statistics=statistics, samples=samples.tolist(), env_data=env_data.tolist())


# %%
def _result_file_name(period_length: int) -> str:
    """Generate the file name from the period length."""
    return f"n_sample_per_period_{period_length}"


def collect_or_calculate_results(period_length: int, num_estimates: int = 2_000) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a saved result for the desired length of time if available, otherwise calculate the result.

    New results will also be saved within this directory.

    Args:
        Args:
        period_length: The number of samples the create a single period of the ERD
        num_estimates: The number of brute force estimates of the QoI. A new period is drawn for each estimate.

    Returns:
        Tuple of:
            ERD samples: (num_estimates,) samples of the ERD for that period length. QoIs can be calculated from this.
            X_max: (num_estimates, d) The location in the environments space that produced the ERD sample.
    """
    results_path = _results_dir / f"{_result_file_name(period_length)}.json"

    samples = torch.tensor([])
    max_location = torch.tensor([])

    if results_path.exists():
        with results_path.open() as fp:
            results = json.load(fp)
            samples = torch.tensor(results["samples"])
            max_location = torch.tensor(results["env_data"])

    # make any additional samples required
    if len(samples) < num_estimates:
        new_samples, new_max_location = brute_force(period_length, num_estimates - len(samples))

        samples = torch.concat([samples, new_samples])
        max_location = torch.concat([max_location, new_max_location])

        # save results:
        with results_path.open("w") as fp:
            json.dump(asdict(ResultsObject.from_samples(samples, max_location)), fp)

    elif len(samples) > num_estimates:
        samples = samples[:num_estimates]
        max_location = max_location[:num_estimates]

    return samples, max_location


def brute_force(period_length: int, num_estimates: int = 2_000) -> tuple[torch.Tensor, torch.Tensor]:
    """Produces brute force samples of the Extreme Response Distribution.

    Args:
        period_length: The number of samples the create a single period of the ERD
        num_estimates: The number of brute force estimates of the QoI. A new period is drawn for each estimate.

    Returns:
        The QoI values calculated for each period. Shape (num_estimates,)
    """
    data: NDArray[np.float64] = np.load(Path(__file__).parent / "data" / "environment_distribution.npy")
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
        dataloader: The dataloader to use to get the environment samples.
             - Each batch should have shape (batch_size, d)
             - The sum of the batch sizes returned by iterating through the dataloader should be a period length
             - To get different results for each brute force estimate, the dataloader needs to give different
               data each time it is iterated through. This can be done by using e.g a RandomSampler.
        num_estimates: The number of brute force estimates of the QoI. A new period is drawn for each estimate.

    Returns:
        Tuple of:
            ERD samples: (num_estimates,) samples of the ERD for that period length. QoIs can be calculated from this.
            X_max: (num_estimates, d) The location in the environments space that produced the ERD sample.
    """
    maxs = torch.zeros(num_estimates)

    _, d = next(iter(dataloader))[0].shape
    maxs_location = torch.zeros(num_estimates, d)

    for i in tqdm.tqdm(range(num_estimates)):
        current_max = float("-inf")

        for batch in dataloader:
            samples = batch[0].to("cpu").numpy()
            loc = _true_loc_func(samples)
            scale = _true_scale_func(samples)

            gumbel_samples: np.ndarray[tuple[int,], Any] = gumbel_r.rvs(loc=loc, scale=scale)  # type: ignore  # noqa: PGH003

            simulator_samples_max = gumbel_samples.max()
            if simulator_samples_max > current_max:
                current_max = simulator_samples_max

                # Get env data corresponding to max(c_max)
                max_index = np.argmax(gumbel_samples)
                maxs_location[i] = torch.tensor(samples[max_index, :])

        maxs[i] = current_max

    return maxs, maxs_location


# %% If you want to run locally to start and save results.
if __name__ == "__main__":
    # To prevent circular imports we redefine the period length here.
    # This should match the value defined in problem.py
    PERIOD_LENGTH = 10000
    # %% Plot the brute force results distribution for the given period length
    samples, x_max = collect_or_calculate_results(PERIOD_LENGTH, 100_000)

    _ = plt.hist(samples, bins=100, density=True)
    _ = plt.title(
        f"Extreme response distribution\n(each result represents the largest value seen {PERIOD_LENGTH} length period)"
    )
    _ = plt.xlabel("Response size")
    _ = plt.ylabel("Density")
    plt.grid(True)  # noqa: FBT003
    plt.savefig(
        f"results/brute_force/erd_n_sample_per_period_{PERIOD_LENGTH}.png",
    )
    plt.show()

    _ = plt.scatter(x_max[:, 0], x_max[:, 1])
    plt.show()

    # %%
    # Here we explore the noise in the brute force result as more samples are collected. We use different sample sizes
    # to get a sense of how the noise in the estimate changes.

    results = []
    brute_force_num_samples = [1_000, 2_000, 4_000, 8_000, 16_000]
    for n_samples in brute_force_num_samples:
        medians_from_samples_size = []
        # How many times to calc the median
        for _ in range(200):
            # sample with replacement
            random_indices = torch.randint(0, len(samples), (n_samples,))
            sampled_tensor = samples[random_indices]
            medians_from_samples_size.append(sampled_tensor.median())

        results.append(torch.tensor(medians_from_samples_size))
    # %%
    # Plot the results
    _, axes = plt.subplots(len(brute_force_num_samples), 1, figsize=(6, len(brute_force_num_samples * 4)), sharex=True)
    for medians, sample_size, ax in zip(results, brute_force_num_samples, axes, strict=True):
        ax.hist(medians, density=True, bins=len(medians) // 15)
        ax.set_title(
            f"QOI calculated with {sample_size} erd samples\nmean (of medians)"
            f" {medians.mean():.3f}. std {medians.std():.3f}"
        )
