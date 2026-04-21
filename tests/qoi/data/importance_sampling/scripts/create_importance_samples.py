"""Generate importance samples and weights for the importance sampling test.

This script creates the importance samples and weights used in
``test_marginal_cdf_extrapolation.test_system_marginal_cdf_with_importance_sampling``.

The importance sampling uses ``importance_sampling_distribution_uniform_region`` which:
1. Draws ``num_samples_total=10_000`` uniform samples from the bounding region ``[[0, 0], [2, 2]]``.
2. Filters to keep only samples where ``env_pdf(x) > threshold (=1e-3)``.
3. Computes importance weights as ``w(x) = env_pdf(x) / h(x)`` where ``h(x)`` is the
   estimated uniform importance PDF over the filtered sub-region.

The environment PDF is ``MultivariateNormal(mean=[0.1, 0.1], cov=0.2*I)`` — the same
distribution used to generate the environment data.
"""

# %%
import sys
from pathlib import Path

import torch

# Add the crest_heights_north_sea example to import the importance sampling function
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[4]
sys.path.insert(0, str(_REPO_ROOT / "examples" / "crest_heights_north_sea"))
sys.path.insert(0, str(_SCRIPT_DIR))
from create_environment import env_pdf
from importance_sampling import importance_sampling_distribution_uniform_region  # type: ignore[import-not-found]

torch.set_default_dtype(torch.float64)

# Importance sampling parameters
REGION = torch.tensor([[0.5, 0.5], [2.0, 2.0]])
THRESHOLD = 1e-3
NUM_SAMPLES_TOTAL = 10_000
SEED = 42

OUTPUT_DIR = Path(__file__).parent.parent


def generate_importance_samples(
    seed: int = SEED,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate importance samples and weights.

    Uses uniform sampling within the region, filtering by the environment PDF threshold,
    and computing weights as ``env_pdf(x) / h(x)``.

    Args:
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (importance_samples, importance_weights).
    """
    _ = torch.manual_seed(seed)

    samples, weights = importance_sampling_distribution_uniform_region(
        env_distribution_pdf=env_pdf,
        region=REGION,
        threshold=THRESHOLD,
        num_samples_total=NUM_SAMPLES_TOTAL,
    )

    return samples, weights


# %%
if __name__ == "__main__":
    # %%
    samples, weights = generate_importance_samples()

    print(f"Samples shape: {samples.shape}")
    print(f"Samples range: [{samples.min():.4f}, {samples.max():.4f}]")
    print(f"Weights shape: {weights.shape}")
    print(f"Weights range: [{weights.min():.6f}, {weights.max():.6f}]")
    print(f"Weights sum: {weights.sum():.4f}")

    samples_path = OUTPUT_DIR / "importance_samples.pt"
    weights_path = OUTPUT_DIR / "importance_weights.pt"

    # %% Save the samples and weights
    torch.save(samples, samples_path)
    torch.save(weights, weights_path)
    print(f"\nSaved to {OUTPUT_DIR}")

    # %%
    # Plot the samples colored by weights to visualize the importance sampling distribution
    # Compare the existing ones with the newly generated ones next to each other
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 1, figsize=(5, 5))
    _ = axes.scatter(samples[:, 0], samples[:, 1], c=weights, cmap="viridis", s=10)
    _ = axes.set_title("Generated Importance Samples")
    _ = axes.set_xlim(REGION[0, 0], REGION[1, 0])
    _ = axes.set_ylim(REGION[0, 1], REGION[1, 1])
    _ = fig.colorbar(plt.cm.ScalarMappable(cmap="viridis"), ax=axes, label="Importance Weight")


# %%
