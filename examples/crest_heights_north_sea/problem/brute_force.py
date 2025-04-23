"""Obtain a brute force estimate of the Extreme Response Distribution (ERD)."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import tqdm
from numpy.typing import NDArray
from simulator import max_crest_height_simulator_function  # type: ignore[import]
from torch import Tensor
from torch.utils.data import DataLoader, RandomSampler, TensorDataset

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


def collect_or_calculate_results(
    n_years_in_period: int,
    n_sea_states_in_year: int,
    num_estimates: int = 2_000,
    year_return_value: int = 100,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return a saved result for the desired length of time if available, otherwise calculate the result.

    New results will also be saved within this directory.

    Args:
        n_years_in_period: number of years to simulate
        n_sea_states_in_year: number of sea states in one year
        num_estimates: The number of brute force estimates of the QoI. A new period is drawn for each estimate.
        year_return_value: R-year return value

    Returns:
        The QoI values calculated for each period. Shape (num_estimates,)
    """
    results_path = (
        _results_dir / f"brut_force_{int(n_years_in_period)}_year_sim_{year_return_value}_year_return_value.json"
    )

    samples = torch.tensor([])

    if results_path.exists():
        with results_path.open() as fp:
            results = json.load(fp)
            samples = torch.tensor(results["samples"])

    # make any additional samples required
    if len(samples) < num_estimates:
        new_samples = brute_force(
            year_return_value * n_sea_states_in_year,
            n_years_in_period,
            num_estimates,
        )

        samples = torch.concat([samples, new_samples])
        # save results
        with results_path.open("w") as fp:
            json.dump(asdict(ResultsObject.from_samples(samples)), fp)
    elif len(samples) > num_estimates:
        samples = samples[:num_estimates]

    return samples, samples.mean(), samples.var()


def brute_force(period_length: int, n_years_in_period: int, num_estimates: int = 2_000) -> torch.Tensor:
    """Produces brute force samples of the Extreme Response Distibtuion.

    Args:
        period_length: The number of samples the create a single period of the ERD
        n_years_in_period: The number of years used to create the environment data. Only needed to load the
        correct dataset.
        num_estimates: The number of brute force estimates of the QoI. A new period is drawn for each estimate.

    Returns:
        The QoI values calculated for each period. Shape (num_estimates,)
    """
    data: NDArray[np.float64] = np.load(
        Path(__file__).parent / "data" / f"long_term_distribution_{n_years_in_period}_years.npy"
    )
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
) -> Tensor:
    """Calculate the QOI by brute force by splitting the data into batches.

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
    maxs = torch.zeros(num_estimates)
    for i in tqdm.tqdm(range(num_estimates)):
        current_max = float("-inf")

        # Get max(c_max) for return period which is specified in dataloader
        for batch in dataloader:
            samples = batch[0].to("cpu").numpy()

            simulator_samples: np.ndarray[tuple[int,], Any] = max_crest_height_simulator_function(samples)  # type: ignore  # noqa: PGH003
            current_max = max(current_max, simulator_samples.max())

        maxs[i] = current_max

    return maxs
