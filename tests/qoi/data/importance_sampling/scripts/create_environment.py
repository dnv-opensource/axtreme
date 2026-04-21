"""Generate the environment distribution for the importance sampling test.

This script creates the environment data used in
``test_marginal_cdf_extrapolation.test_system_marginal_cdf_with_importance_sampling``.

The environment distribution is ``MultivariateNormal(mean=[0.1, 0.1], cov=0.2*I)``,
truncated to positive values. This places the bulk of environment samples away from the
extreme response region (which peaks at [1, 1]), so that only few environment samples cover
the subregion where the extreme response occurs. This motivates the use of importance sampling.

Usage::

    python create_environment.py

Output:
    ``tests/qoi/data/importance_sampling/environment_distribution.npy``
    Shape: (10000, 2), dtype: float64
"""

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.distributions import MultivariateNormal

torch.set_default_dtype(torch.float64)

# Environment distribution: MultivariateNormal(mean=[0.1, 0.1], cov=diag(0.2))
# Samples are filtered to keep only positive values in both dimensions.
ENV_MEAN = torch.tensor([0.1, 0.1])
ENV_COV = torch.tensor([[0.2, 0], [0, 0.2]])

N_SAMPLES = 10_000
SEED = 42

OUTPUT_DIR = Path(__file__).parent.parent


def generate_environment_data(
    n_samples: int = N_SAMPLES,
    seed: int = SEED,
) -> np.ndarray:
    """Generate environment samples from a truncated MultivariateNormal.

    Samples from ``MultivariateNormal(mean=[0.1, 0.1], cov=0.2*I)`` and discards any
    samples with negative values.

    Args:
        n_samples: Number of samples to generate.
        seed: Random seed for reproducibility.

    Returns:
        Array of shape (n_samples, 2) of environment samples.
    """
    env_mvn = MultivariateNormal(ENV_MEAN, covariance_matrix=ENV_COV)

    # This was how the samples were originally generated but I have decided to keep all samples to prevent any issues
    # with the brute force
    # with torch.random.fork_rng():
    #     _ = torch.manual_seed(seed)
    #     valid_samples: list[np.ndarray] = []

    #     while len(valid_samples) < n_samples:
    #         samples = env_mvn.sample(torch.Size([n_samples]))
    #         mask = (samples >= 0).all(dim=1)
    #         batch_valid = samples[mask].numpy()
    #         for sample in batch_valid:
    #             if len(valid_samples) < n_samples:
    #                 valid_samples.append(sample)
    #             else:
    #                 break

    return np.array(env_mvn.sample(torch.Size([n_samples])))  # np.array(valid_samples[:n_samples])


def env_pdf(x: torch.Tensor) -> torch.Tensor:
    """Evaluate the environment PDF at given points.

    This is the PDF of the underlying (untruncated) MultivariateNormal. This is passed to the
    importance sampling algorithm as ``env_distribution_pdf``.

    Args:
        x: Tensor of shape (n_points, 2).

    Returns:
        Tensor of shape (n_points,) with PDF values.
    """
    env_mvn = MultivariateNormal(ENV_MEAN, covariance_matrix=ENV_COV)
    return torch.exp(env_mvn.log_prob(x))


# %%
if __name__ == "__main__":
    # %%
    env_data = generate_environment_data()

    print(f"Shape: {env_data.shape}")
    print(f"Mean: {env_data.mean(axis=0)}")
    print(f"Std: {env_data.std(axis=0)}")
    print(f"Min: {env_data.min(axis=0)}")
    print(f"Max: {env_data.max(axis=0)}")

    # %% Save the environment data
    output_path = OUTPUT_DIR / "environment_distribution.npy"
    np.save(output_path, env_data)
    print(f"\nSaved to {output_path}")

    # %% Plotting the generated data to visualize the distribution

    _ = plt.figure(figsize=(6, 6))
    _ = plt.scatter(env_data[:, 0], env_data[:, 1], alpha=0.5, s=10)
    _ = plt.title("Generated Environment Samples")

# %%
