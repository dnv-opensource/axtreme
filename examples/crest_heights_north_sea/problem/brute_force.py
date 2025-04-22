"""Obtain a brute force estimate of the Extreme Response Distribution (ERD)."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import tqdm
from simulator import max_crest_height_simulator_function  # type: ignore[import]
from torch import Tensor
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

    @classmethod
    def from_samples(cls, samples: torch.Tensor) -> "ResultsObject":
        """Create the object directly from samples."""
        statistics = {"median": float(samples.median()), "mean": float(samples.mean())}
        return ResultsObject(statistics=statistics, samples=samples.tolist())


def _result_file_name(period_length: int) -> str:
    """Generate the file name from the period length."""
    return f"n_sample_per_period_{period_length}"


def collect_or_calculate_results(
    dataloader: DataLoader[tuple[Tensor, ...]],
    n_sea_states_in_year: int,
    n_sea_states_in_period: int,
    num_estimates: int = 2_000,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return a saved result for the desired length of time if available, otherwise calculate the result.

    New results will also be saved within this directory.

    Args:
        dataloader: Dataloader objective of environment data
        n_sea_states_in_year: number of sea states in one year
        n_sea_states_in_period: number of sea states in whole period
        num_estimates: The number of brute force estimates of the QoI. A new period is drawn for each estimate.

    Returns:
        The QoI values calculated for each period. Shape (num_estimates,)
    """
    # Calculate the number of samples that create a single period of the ERD
    period_length = dataloader.dataset.data.shape[0]  # type: ignore[attr-defined]
    results_path = _results_dir / f"{_result_file_name(period_length)}.json"

    samples = torch.tensor([])

    if results_path.exists():
        with results_path.open() as fp:
            results = json.load(fp)
            samples = torch.tensor(results["samples"])

    # to reduce run time for testing
    year_return_value = 10

    # make any additional samples required
    if len(samples) < num_estimates:
        new_samples = brute_force_calc(
            dataloader,
            n_sea_states_in_period,
            n_sea_states_in_year,
            num_estimates - len(samples),
            year_return_value,
        )

        samples = torch.concat([samples, new_samples])
        # save results
        with results_path.open("w") as fp:
            json.dump(asdict(ResultsObject.from_samples(samples)), fp)
    elif len(samples) > num_estimates:
        samples = samples[:num_estimates]

    return samples, samples.mean(), samples.var()


def brute_force_calc(
    dataloader: DataLoader[tuple[Tensor, ...]],
    n_sea_states_in_period: int,
    n_sea_states_in_year: int = 2922,
    num_estimates: int = 2_000,
    year_return_value: int = 100,
) -> Tensor:
    """Calculate the QOI by brute force by splitting the data into year_return_value chuncks.

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
    for _ in tqdm.tqdm(range(num_estimates)):
        chunck_indices = np.append(
            np.arange(0, n_sea_states_in_period, year_return_value * n_sea_states_in_year), n_sea_states_in_period
        )

        for idx, chunck_index in enumerate(chunck_indices[1:]):
            samples = dataloader.dataset.data[chunck_indices[idx] : chunck_index, :]  # type: ignore[attr-defined]

            simulator_samples: np.ndarray[tuple[int,], Any] = max_crest_height_simulator_function(samples)  # type: ignore  # noqa: PGH003
            maxs.append(simulator_samples.max())

    return torch.FloatTensor(maxs)
