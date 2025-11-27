"""Handle the collection of data."""

# %%
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.distributions import MultivariateNormal


def generate_and_save_data(n_samples: int = 1_000_000, seed: int = 42) -> None:
    """Generate environment data using multivariate normal distribution and save to file.

    Args:
        n_samples: Number of samples to generate (default: 1 000 000).
        seed: Random seed for reproducibility (default: 42).
    """
    mean = torch.tensor([0.4, 0.4])
    cov = torch.tensor([[0.2, 0], [0, 0.2]])

    # Create distribution and sample
    env_mvn = MultivariateNormal(mean, covariance_matrix=cov)
    with torch.random.fork_rng():
        _ = torch.manual_seed(seed)
        # Generate samples until we have enough valid ones
        valid_samples: list[np.ndarray] = []  # type: ignore  # noqa: PGH003

        while len(valid_samples) < n_samples:
            samples_tensor = env_mvn.sample(torch.Size([n_samples]))
            samples_np = samples_tensor.numpy()

            # Filter samples to [0, 1] range
            mask = (samples_np >= 0) & (samples_np <= 1)
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


# %%
if __name__ == "__main__":
    generate_and_save_data(n_samples=50_000, seed=42)

# %%
