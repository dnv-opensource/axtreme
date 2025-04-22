"""Obtain a brute force estimate of the Extreme Response Distribution (ERD)."""

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
from simulator import max_crest_height_simulator_function  # type: ignore[import]
from torch import Tensor, zeros
from torch.utils.data import DataLoader

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


def _result_file_name(period_length: int) -> str:
    """Generate the file name from the period length."""
    return f"n_sample_per_period_{period_length}"


def collect_or_calculate_results(
    dataloader: DataLoader[tuple[Tensor, ...]],
    n_sea_states_in_year: int,
    n_sea_states_in_period: int,
    num_estimates: int = 2_000,
    brut_force_type: str = "quantile",
    year_return_value: int = 10,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return a saved result for the desired length of time if available, otherwise calculate the result.

    New results will also be saved within this directory.

    Args:
        dataloader: Dataloader objective of environment data
        n_sea_states_in_year: number of sea states in one year
        n_sea_states_in_period: number of sea states in whole period
        num_estimates: The number of brute force estimates of the QoI. A new period is drawn for each estimate.
        brut_force_type: Choose how the QoI shall be estimated.
        year_return_value: return value given in years

    Returns:
        The QoI values calculated for each period. Shape (num_estimates,)
    """
    # Calculate the number of samples that create a single period of the ERD
    period_length = dataloader.dataset.data.shape[0]  # type: ignore[attr-defined]
    results_path = (
        _results_dir / f"{_result_file_name(period_length)}_{brut_force_type}_{year_return_value}_return_year.json"
    )

    samples = torch.tensor([])
    max_location = torch.tensor([])

    if results_path.exists():
        with results_path.open() as fp:
            results = json.load(fp)
            samples = torch.tensor(results["samples"])
            max_location = torch.tensor(results["env_data"])

    # make any additional samples required
    if len(samples) < num_estimates:
        if brut_force_type == "quantile":
            new_samples = quantile_brute_force_calc(
                dataloader,
                n_sea_states_in_year,
                num_estimates - len(samples),
                year_return_value,
            )
        elif brut_force_type == "chunck":
            new_samples, new_max_location = chunck_brute_force_calc(
                dataloader,
                n_sea_states_in_period,
                n_sea_states_in_year,
                num_estimates - len(samples),
                year_return_value,
            )

        samples = torch.concat([samples, new_samples])
        max_location = torch.concat([max_location, new_max_location])

        # save results
        with results_path.open("w") as fp:
            json.dump(asdict(ResultsObject.from_samples(samples, max_location)), fp)
    elif len(samples) > num_estimates:
        samples = samples[:num_estimates]

    return samples, samples.mean(), samples.var()


def quantile_brute_force_calc(
    dataloader: DataLoader[tuple[Tensor, ...]],
    n_sea_states_in_year: int = 2922,
    num_estimates: int = 2_000,
    year_return_value: int = 100,
) -> Tensor:
    """Calculate the QOI by brute force using quantiles.

    Args:
        dataloader: The dataloader to use to get the environment samples.
             - Each batch should have shape (batch_size, d)
             - The sum of the batch sizes returned by iterating through the dataloader should be a period length
             - To get different results for each brute force estimate, the dataloader needs to give different
               data each time it is iterated through. This can be done by using e.g a RandomSampler.
        n_sea_states_in_year: number of sea states in one year
        num_estimates: The number of brute force estimates of the QoI. A new period is drawn for each estimate.
        year_return_value: year for the return value

    Returns:
        The QoI values calculated for each period. Shape (num_estimates,)
    """
    return_value = zeros(num_estimates)
    for i in tqdm.tqdm(range(num_estimates)):
        simulator_samples: np.ndarray[tuple[int,], Any] = max_crest_height_simulator_function(dataloader.dataset.data)  # type: ignore  # noqa: PGH003

        return_value[i] = float(np.quantile(simulator_samples, 1 - 1 / (n_sea_states_in_year * year_return_value)))

    return return_value


def chunck_brute_force_calc(
    dataloader: DataLoader[tuple[Tensor, ...]],
    n_sea_states_in_period: int,
    n_sea_states_in_year: int = 2922,
    num_estimates: int = 2_000,
    year_return_value: int = 100,
) -> tuple[Tensor, Tensor]:
    """Calculate the QOI by brute force by splitting the data into year_return_value chunck.

    Note: At the meeting with Odin on the 22.04.25 it was decided that this is the appropriate method.

    Args:
        dataloader: The dataloader to use to get the environment samples.
             - Each batch should have shape (batch_size, d)
             - The sum of the batch sizes returned by iterating through the dataloader should be a period length
             - To get different results for each brute force estimate, the dataloader needs to give different
               data each time it is iterated through. This can be done by using e.g a RandomSampler.
        n_sea_states_in_year: number of sea states in one year
        n_sea_states_in_period: number of sea states in whole period
        num_estimates: The number of brute force estimates of the QoI. A new period is drawn for each estimate.
        year_return_value: year for the return value

    Returns:
        The QoI values calculated for each period. Shape (num_estimates,)
    """
    maxs = []
    maxs_location = []
    for _ in tqdm.tqdm(range(num_estimates)):
        chunck_indices = np.append(
            np.arange(0, n_sea_states_in_period, year_return_value * n_sea_states_in_year), n_sea_states_in_period
        )

        for idx, chunck_index in enumerate(chunck_indices[1:]):
            samples = dataloader.dataset.data[chunck_indices[idx] : chunck_index, :]  # type: ignore[attr-defined]

            simulator_samples: np.ndarray[tuple[int,], Any] = max_crest_height_simulator_function(samples)  # type: ignore  # noqa: PGH003
            maxs.append(simulator_samples.max())

            # Get env data corresponding to max(c_max)
            max_index = np.argmax(simulator_samples)
            maxs_location.append(samples[max_index, :])

    return torch.FloatTensor(maxs), torch.FloatTensor(maxs_location)


class FileNotFoundCustomError(Exception):
    """Exception raised when the brute force file is not found."""


def create_extrem_value_location_scatter_plot(brut_force_file_name: str) -> None:
    """Make scatter plot of the extrem value location.

    The plot shows where in the (Hs, Tp) the maxima of c_max occured
    when using the brut force approach.

    Args:
        brut_force_file_name: file name where the brutforce results are stored.
    """
    brut_force_file_path = _results_dir / brut_force_file_name
    if brut_force_file_path.exists():
        with brut_force_file_path.open() as fp:
            results = json.load(fp)
            max_location = torch.tensor(results["env_data"])
    else:
        raise FileNotFoundCustomError(f"File {brut_force_file_path} not found.")

    _ = plt.scatter(max_location[:, 0], max_location[:, 1], s=1, alpha=0.5)
    _ = plt.title("Extrem value location")  # type: ignore[assignment]
    _ = plt.xlabel("Hs")  # type: ignore[assignment]
    _ = plt.ylabel("Tp")  # type: ignore[assignment]
    plt.grid(True)  # noqa: FBT003

    plt.savefig(str(brut_force_file_path).replace(".json", "_scatter.png"))


def create_extrem_value_location_kde_plot(brut_force_file_name: str) -> None:
    """Make KDE (kernel density estimate) plot of the extrem value location.

    The plot shows where in the (Hs, Tp) the maxima of c_max occured
    when using the brut force approach.

    Args:
        brut_force_file_name: file name where the brutforce results are stored.
    """
    brut_force_file_path = _results_dir / brut_force_file_name
    if brut_force_file_path.exists():
        with brut_force_file_path.open() as fp:
            results = json.load(fp)
            max_location = pd.DataFrame(results["env_data"], columns=["Hs", "Tp"])
    else:
        raise FileNotFoundCustomError(f"File {brut_force_file_path} not found.")

    _ = sns.kdeplot(
        data=max_location,
        x="Hs",
        y="Tp",
        fill=True,
    )
    _ = plt.title("Extrem value location")  # type: ignore[assignment]
    _ = plt.xlabel("Hs")  # type: ignore[assignment]
    _ = plt.ylabel("Tp")  # type: ignore[assignment]

    plt.savefig(str(brut_force_file_path).replace(".json", "_kde.png"))
