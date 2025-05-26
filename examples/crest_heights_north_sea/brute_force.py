"""Obtain a brute force estimate of the extreme Response Distribution (ERD)."""

# %%
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import tqdm
from numpy.typing import NDArray
from simulator import max_crest_height_simulator_function  # type: ignore[import]
from torch import Tensor
from torch.utils.data import DataLoader, RandomSampler, TensorDataset

_results_dir: Path = Path(__file__).parent / "results" / "brute_force"
if not _results_dir.exists():
    _results_dir.mkdir(parents=True, exist_ok=True)

_: Any


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
        statistics = {"median": float(samples.median()), "mean": float(samples.mean()), "n_samples": samples.shape[0]}
        return ResultsObject(statistics=statistics, samples=samples.tolist(), env_data=env_data.tolist())


def collect_or_calculate_results(
    period_length: int,
    num_estimates: int = 2_000,
) -> tuple[Tensor, Tensor]:
    """Return a saved result for the desired period length if available, otherwise calculate the result.

    New results will also be saved to json.

    Args:
        period_length: The number of environment samples that create a single period of the ERD
        num_estimates: The number of ERD samples to create. A new period is drawn for each estimate.

    Returns:
        Tuple of:
            ERD samples: (num_estimates,) samples of the ERD for that period length. QoIs can be calculated from this.
            X_max: (num_estimates, d) The location in the environments space that produced the ERD sample.
    """
    results_path = _results_dir / f"{int(period_length)}_period_length.json"

    samples = torch.tensor([])
    max_location = torch.tensor([])

    if results_path.exists():
        with results_path.open() as fp:
            results = json.load(fp)
            samples = torch.tensor(results["samples"])
            max_location = torch.tensor(results["env_data"])

    # make any additional samples required
    if len(samples) < num_estimates:
        new_samples, new_max_location = brute_force(
            period_length,
            num_estimates - len(samples),
        )
        samples = torch.concat([samples, new_samples])
        max_location = torch.concat([max_location, new_max_location])

        # save results
        with results_path.open("w") as fp:
            json.dump(asdict(ResultsObject.from_samples(samples, max_location)), fp)
    elif len(samples) > num_estimates:
        samples = samples[:num_estimates]
        max_location = max_location[:num_estimates]

    return samples, max_location


def brute_force(period_length: int, num_estimates: int = 2_000) -> tuple[Tensor, Tensor]:
    """Produces brute force samples of the Extreme Response Distribution.

    Args:
        period_length: The number of samples that create a single period of the ERD
        num_estimates: The number of brute force estimates of the QoI. A new period is drawn for each estimate.

    Returns:
        Tuple of:
            ERD samples: (num_estimates,) samples of the ERD for that period length. QoIs can be calculated from this.
            X_max: (num_estimates, d) The location in the environments space that produced the ERD sample.
    """
    data: NDArray[np.float64] = np.load(Path(__file__).parent / "usecase" / "data" / "long_term_distribution.npy")
    dataset = TensorDataset(torch.Tensor(data))

    dataloader = DataLoader(
        dataset,
        batch_size=4096,
        sampler=RandomSampler(dataset, num_samples=period_length, replacement=True),
    )

    return _brute_force_calc(dataloader, num_estimates)


def _brute_force_calc(
    dataloader: DataLoader[tuple[Tensor, ...]],
    num_estimates: int = 2_000,
) -> tuple[Tensor, Tensor]:
    """Generate samples of the ERD via brute force.

    Run the simulator on many periods of environment data, extract the max value for each period.

    Args:
        dataloader: The dataloader to use to get the environment samples.
             - Each batch should have shape (batch_size, d)
             - The sum of the batch sizes returned by iterating through the dataloader should be a return period
             - To get different results for each brute force estimate, the dataloader needs to give different
               data each time it is iterated through. This can be done by using e.g a RandomSampler.
             - dataset is expected to be a TensorDataset (which wraps index results in a tuple)
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

        # Get max(c_max) for return period which is specified in dataloader
        for batch in dataloader:
            samples = batch[0].to("cpu").numpy()

            simulator_samples: np.ndarray[tuple[int, int], np.dtype[np.float64]] = max_crest_height_simulator_function(
                samples
            )

            simulator_samples_max = simulator_samples.max()
            if simulator_samples_max > current_max:
                current_max = simulator_samples_max

                # Get env data corresponding to max(c_max)
                max_index = np.argmax(simulator_samples)
                maxs_location[i] = torch.tensor(samples[max_index, :])

        maxs[i] = current_max

    return maxs, maxs_location


def create_extreme_value_location_scatter_plot(brute_force_file_name: str) -> None:
    """Make scatter plot of the extreme value location.

    The plot shows where in the (Hs, Tp) the maxima of c_max occurred when using the brute force approach.

    Args:
        brute_force_file_name: file name where the brute force results are stored.
    """
    brute_force_file_path = _results_dir / brute_force_file_name

    with brute_force_file_path.open("r") as fp:
        results = json.load(fp)
        max_location = torch.tensor(results["env_data"])

    _ = plt.scatter(max_location[:, 0], max_location[:, 1], s=1, alpha=0.5)
    _ = plt.title("extreme value location")
    _ = plt.xlabel("Hs")
    _ = plt.ylabel("Tp")
    plt.grid(True)  # noqa: FBT003

    plt.savefig(str(brute_force_file_path).replace(".json", "_scatter.png"))


def create_extreme_value_location_kde_plot(brute_force_file_name: str) -> None:
    """Make KDE (kernel density estimate) plot of the extreme value location.

    The plot shows where in the (Hs, Tp) the maxima of c_max occurred when using the brute force approach.

    Args:
        brute_force_file_name: file name where the brute force results are stored.
    """
    brute_force_file_path = _results_dir / brute_force_file_name

    with brute_force_file_path.open("r") as fp:
        results = json.load(fp)
        max_location = pd.DataFrame(results["env_data"], columns=["Hs", "Tp"])

    _ = sns.kdeplot(
        data=max_location,
        x="Hs",
        y="Tp",
        fill=True,
    )
    _ = plt.title("extreme value location")
    _ = plt.xlabel("Hs")
    _ = plt.ylabel("Tp")

    plt.savefig(str(brute_force_file_path).replace(".json", "_kde.png"))


# %% The following produces and analyses the brute fore estimate
if __name__ == "__main__":
    # Set parameters for simulation
    year_return_value = 10
    n_sea_states_in_year = 2922
    period_length = year_return_value * n_sea_states_in_year

    # %%
    # Get brute force QOI for a large number of estimates
    extreme_response_values, extreme_response_env = collect_or_calculate_results(
        period_length,
        num_estimates=100_001,
    )

    # %%
    print(f"brute force exp(-1) quantile: {torch.quantile(extreme_response_values, np.exp(-1))}")

    # %%
    # Plot brute force QOI
    _ = plt.hist(extreme_response_values, bins=100, density=True)
    _ = plt.title("R-year return value distribution")
    _ = plt.xlabel("R-year return value")
    _ = plt.ylabel("Density")
    _ = plt.axvline(extreme_response_values.mean().item(), color="red", label="mean")
    _ = plt.axvline(extreme_response_values.median().item(), color="purple", label="median")
    _ = plt.axvline(
        torch.quantile(extreme_response_values, np.exp(-1)).item(), color="orange", label="exp(-1) quantile"
    )
    _ = plt.legend()
    plt.grid(True)  # noqa: FBT003

    # %% [markdown]
    # We need to estimate the amount of uncertainty there is in the point estimate. The following shows a negligible
    #  amount when 16_000 samples are used (Obviously these are not truly random samples so this is not perfect).
    # %%
    # Analyse uncertainty in brute force estimate using both median and exp(-1) quantile
    results_quantile = []
    brute_force_samples = [1_000, 2_000, 4_000, 8_000, 16_000]
    for n_samples in brute_force_samples:
        quantiles_from_samples_size = []
        # How many times to calc the median
        for _idx in range(200):
            # sample with replacement
            random_indices = torch.randint(0, len(extreme_response_values), (n_samples,))
            sampled_tensor = extreme_response_values[random_indices]
            quantiles_from_samples_size.append(torch.quantile(sampled_tensor, np.exp(-1)))

        results_quantile.append(torch.tensor(quantiles_from_samples_size))

    # %% Plot the results using the quantile
    fig, axes = plt.subplots(len(brute_force_samples), 1, figsize=(6, len(brute_force_samples * 4)), sharex=True)
    for quantile, sample_size, ax in zip(results_quantile, brute_force_samples, axes, strict=True):
        ax.hist(quantile, density=True, bins=len(quantile) // 15)
        ax.set_title(
            f"QOI calculated with {sample_size} erd samples\nmean (of exp(-1) quantile)"
            f" {quantile.mean():.3f}. std {quantile.std():.3f}"
        )
    # %% Plot scatter plot of brute force solution for extreme value location
    create_extreme_value_location_scatter_plot(f"{int(period_length)}_period_length.json")

    # %% Plot kde plot of brute force solution for extreme value location
    # Note: Is very slow especially for large datasets
    create_extreme_value_location_kde_plot(f"{int(period_length)}_period_length.json")


# %%
