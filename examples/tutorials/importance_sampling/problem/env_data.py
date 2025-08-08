"""Handle the collection of data."""

# %%
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.distributions import MultivariateNormal


def define_mean_cov_for_env_distribution() -> tuple[torch.Tensor, torch.Tensor]:
    """Define mean and covariance in one place."""
    return torch.tensor([0.1, 0.1]), torch.tensor([[0.2, 0], [0, 0.2]])


def generate_and_save_data(
    n_samples: int = 10_000, seed: int = 42, value_range_start: float = 0, value_range_end: float = 1
) -> None:
    """Generate environment data using a multivariate normal distribution and save it to a file.

    Args:
        n_samples: Number of samples to generate (default: 10 000).
        seed: Random seed for reproducibility (default: 42).
        value_range_start: All env samples shall be larger or equal than this value (default: 0).
        value_range_end: All env samples shall be smaller or equal than this value (default: 1)
    """
    mean, cov = define_mean_cov_for_env_distribution()

    # Create distribution and sample
    env_mvn = MultivariateNormal(mean, covariance_matrix=cov)
    with torch.random.fork_rng():
        _ = torch.manual_seed(seed)
        # Generate samples until we have enough valid ones
        valid_samples: list[np.ndarray] = []  # type: ignore  # noqa: PGH003

        while len(valid_samples) < n_samples:
            samples_tensor = env_mvn.sample(torch.Size([n_samples]))
            samples_np = samples_tensor.numpy()

            # Filter samples to [0, 2] range
            mask = (samples_np >= value_range_start) & (samples_np <= value_range_end)
            valid_mask = mask.all(axis=1)
            batch_valid = samples_np[valid_mask]

            # Add to our collection
            for sample in batch_valid:
                if len(valid_samples) < n_samples:
                    valid_samples.append(sample)
                else:
                    break
        env_data = np.array(valid_samples[:n_samples])

    # Save to file
    current_dir = Path(__file__).parent
    data_dir = current_dir / "data"
    data_dir.mkdir(exist_ok=True)

    np.save(data_dir / "environment_distribution.npy", env_data)


def collect_data() -> pd.DataFrame:
    """Returns a dataframe of the env data."""
    current_dir = Path(__file__).parent
    data_file = current_dir / "data/environment_distribution.npy"

    # Generate data if the data file does not exist
    if not data_file.exists():
        generate_and_save_data()
        print("Generated and saved data samples to environment_distribution.npy")

    numpy_data = np.load(data_file)
    return pd.DataFrame(numpy_data, columns=["x1", "x2"])


def calculate_environment_distribution(samples: torch.Tensor) -> torch.Tensor:
    """Calculate the probability density function (pdf) for given samples.

    Args:
        samples: Samples for which the pdf should be calculated.
    """
    mean, cov = define_mean_cov_for_env_distribution()
    distribution = MultivariateNormal(mean, covariance_matrix=cov)

    # log_prob returns the log of the pdf evaluated at a value. Pytorch does not provide the pdf
    # directly. Therefore, we need to use the exponential of the log pdf.
    return torch.exp(distribution.log_prob(samples))


# %%
if __name__ == "__main__":
    # %%
    generate_and_save_data(n_samples=50_000, seed=42)

# %%
