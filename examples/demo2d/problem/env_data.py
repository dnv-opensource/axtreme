"""Handle the collection of data."""

# %%
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.distributions import MultivariateNormal


def generate_and_save_data(n_samples: int = 50_000, seed: int = 42) -> None:
    """Generate environment data using multivariate normal distribution and save to file."""
    # Define mean and covariance (using the distribution from junk.py)
    mean = torch.tensor([0.4, 0.4])
    cov = torch.tensor([[0.2, 0], [0, 0.2]])

    # Create distribution and sample
    env_mvn = MultivariateNormal(mean, covariance_matrix=cov)
    with torch.random.fork_rng():
        _ = torch.manual_seed(seed)
        samples = env_mvn.sample(torch.Size([n_samples]))

    # Convert to numpy and filter to [0,1] range
    env_data = samples.numpy()
    env_data = env_data[(env_data > 0).all(axis=1)]  # remove negative values
    env_data = env_data[(env_data < 1).all(axis=1)]  # remove values > 1

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

    numpy_data = np.load(data_file)
    return pd.DataFrame(numpy_data, columns=["x1", "x2"])


# %%
if __name__ == "__main__":
    generate_and_save_data(n_samples=10_000_000, seed=42)
