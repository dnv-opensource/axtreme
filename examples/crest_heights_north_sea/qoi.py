"""Evaluate the convergence of the QOI for different training datasets."""

# %%

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from ax import (
    Experiment,
)
from ax.modelbridge.registry import Models
from numpy.typing import NDArray
from plotly.subplots import make_subplots
from problem import (  # type: ignore[import-not-found]
    DIST,
    SEARCH_SPACE,
    brut_force_qoi,
    dataset,
    hs_bounds,
    period_length,
    sim,
    tp_bounds,
)
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import Tensor
from torch.distributions import Normal
from torch.utils.data import DataLoader, Dataset

from axtreme.data import FixedRandomSampler, ImportanceAddedWrapper
from axtreme.data.dataset import MinimalDataset
from axtreme.experiment import add_sobol_points_to_experiment, make_experiment
from axtreme.plotting.histogram3d import histogram_surface3d
from axtreme.qoi import MarginalCDFExtrapolation
from axtreme.qoi.qoi_estimator import QoIEstimator
from axtreme.sampling.importance_sampling import importance_sampling_distribution_uniform_region
from axtreme.sampling.ut_sampler import UTSampler
from axtreme.utils import population_estimators, transforms


# %%
# Set up experiment
def make_exp() -> Experiment:
    """Convenience function returns a fresh Experiment of this problem."""
    return make_experiment(sim, SEARCH_SPACE, DIST, n_simulations_per_point=1000)


# %% We want to compare the QoI with and without importance sampling

num_samples = 5_000  # number of samples to be used for sampler
seed_random_sampler = 10  # Seed used in random sampler
batch_size = 256  # Batch size for random sampler

posterior_sampler = UTSampler()

quantile = np.exp(-1)  # Quantile for QoI

n_training_points = [30, 50, 100]  # , 200, 500]  # List of number of training points for GP


def run_qoi_estimation(dataloader: DataLoader) -> tuple[list[Tensor], QoIEstimator]:
    """Performs QoI estimation for a given dataloader."""
    qoi_estimator = MarginalCDFExtrapolation(
        env_iterable=dataloader,
        period_len=period_length,
        quantile=torch.tensor(quantile),
        quantile_accuracy=torch.tensor(0.01),
        posterior_sampler=posterior_sampler,
    )

    results = []

    for points in tqdm.tqdm(n_training_points):
        exp = make_exp()
        add_sobol_points_to_experiment(exp, n_iter=points, seed=8)
        # Use ax to create a gp from the experiment
        botorch_model_bridge = Models.BOTORCH_MODULAR(
            experiment=exp,
            data=exp.fetch_data(),
        )

        # We need to collect the transforms used to the model gives result in the problem/outcome space.
        input_transform, outcome_transform = transforms.ax_to_botorch_transform_input_output(
            transforms=list(botorch_model_bridge.transforms.values()), outcome_names=botorch_model_bridge.outcomes
        )
        qoi_estimator.input_transform = input_transform
        qoi_estimator.outcome_transform = outcome_transform

        model = botorch_model_bridge.model.surrogate.model

        results.append(qoi_estimator(model))
    return results, qoi_estimator


# %% Step 1: Run QoI without importance sampling
sampler_whole_dataset = FixedRandomSampler(dataset, num_samples=num_samples, seed=seed_random_sampler, replacement=True)  # type: ignore[arg-type]
dataloader_whole_dataset = DataLoader(dataset, sampler=sampler_whole_dataset, batch_size=batch_size)

results_whole_dataset, qoi_estimator_whole_dataset = run_qoi_estimation(dataloader_whole_dataset)


# Step 2: Run QoI with importance sampling
# %% Step 2a: Get environment distribution
def set_up_env_data_distribution_from_samples(
    env_data: Dataset[NDArray[np.float64]], num_training_samples: int = 1_000
) -> Pipeline:
    """Calculate the probability density function of the environment distribution."""
    pipeline = Pipeline(
        [
            ("normalize", StandardScaler()),
            ("regressor", KernelDensity()),
        ]
    )
    params = {"regressor__bandwidth": np.logspace(-1, 1, 20)}

    # Fit the distribution
    grid = GridSearchCV(pipeline, params, n_jobs=-1)

    # Only use a subset of the data to save time
    samples = torch.tensor(env_data)
    training_samples = samples[torch.randint(0, samples.shape[0], (num_training_samples,))]

    _ = grid.fit(training_samples)
    kde_estimator = grid.best_estimator_

    return kde_estimator


kde_pipeline = set_up_env_data_distribution_from_samples(dataset.data)


# TODO(am-kaiser): Find a more elegant version than passing a global parameter for
# kde_pipeline. The function importance_sampling_distribution_uniform_region only accepts one
# argument for env_distribution_pdf.
def env_distribution_pdf(x: torch.Tensor, kde_pipeline: Pipeline = kde_pipeline) -> torch.Tensor:
    """Calculate the probability density function of the environment distribution."""
    return torch.exp(torch.tensor(kde_pipeline.score_samples(x)))


# %% Step 2b: Create importance samples

# Define parameters for importance sampling
## Threshold, set to near 0 for now as a PDF is never negative
threshold = 1e-10
# Number of samples generated by importance sampling algorithm
num_importance_samples = int(1e5)
# We choose the region to be a hyper rectangle that contains the significant regions of
# the real environment distribution
# Important: values need to be float!
region = torch.tensor([[float(hs_bounds[0]), float(tp_bounds[0])], [float(hs_bounds[1]), float(tp_bounds[1])]])

importance_samples, importance_weights = importance_sampling_distribution_uniform_region(
    env_distribution_pdf, region, threshold, num_importance_samples
)

# Save importance samples and weights
np.save("usecase/results/importance_samples/importance_samples.npy", importance_samples)
np.save("usecase/results/importance_samples/importance_weights.npy", importance_weights)

# Plot the importance sampling samples vs the environment data
fig = make_subplots(
    rows=1,
    cols=2,
    specs=[[{"type": "surface"}, {"type": "surface"}]],
    subplot_titles=["env_data Distribution", "Importance Sampling Distribution"],
)

# Add surface for env_data
surf1 = histogram_surface3d(dataset.data, n_bins=20).data[0]
_ = surf1.update(coloraxis="coloraxis1")
_ = fig.add_trace(surf1, row=1, col=1)

# Add surface for importance_samples
surf2 = histogram_surface3d(importance_samples.numpy(), n_bins=20).data[0]
_ = surf2.update(coloraxis="coloraxis2")
_ = fig.add_trace(surf2, row=1, col=2)

_ = fig.update_layout(
    scene={"aspectmode": "cube"},
    scene2={"aspectmode": "cube"},
    coloraxis1={
        "colorscale": "Viridis",
        "colorbar": {"title": "PDF", "x": 0.40, "xanchor": "left"},
    },
    coloraxis2={
        "colorscale": "Viridis",
        "colorbar": {"title": "PDF", "x": 1, "xanchor": "left"},
    },
)
fig.show()

# %% Step 2c: Run QoI with importance samples
# Combine the importance sampled data and the weights
importance_dataset = ImportanceAddedWrapper(MinimalDataset(importance_samples), MinimalDataset(importance_weights))

# Create new dataloader using the importance sample
importance_sample_sampler = FixedRandomSampler(
    importance_dataset, num_samples=num_samples, replacement=True, seed=seed_random_sampler
)
importance_sample_dataloader = DataLoader(importance_dataset, sampler=importance_sample_sampler, batch_size=batch_size)

results_importance_sample, qoi_estimator_importance_sample = run_qoi_estimation(importance_sample_dataloader)


# %% Compare QoI estimation with and without importance sampling
def get_mean_var(estimator: QoIEstimator, estimates: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Get mean and variance for an estimator.

    Args:
        estimator: the QoI function that produced the estimate
        estimates: (*b, n_estimator)

    Returns:
        tensor1: the mean of the estimates, with shape *b
        tensor1: the mean of the estimates, with shape *b

    """
    if not isinstance(estimates, torch.Tensor):  # pyright: ignore[reportUnnecessaryIsInstance]
        estimates = torch.tensor(estimates)

    mean = estimator.posterior_sampler.mean(estimates, -1)  # type: ignore[attr-defined]
    var = estimator.posterior_sampler.var(estimates, -1)  # type: ignore[attr-defined]

    return mean, var


fig, axes = plt.subplots(nrows=len(n_training_points), sharex=True, figsize=(6, 6 * len(n_training_points)))

for idx, (ax, n_points) in enumerate(zip(axes, n_training_points, strict=True)):
    # Create distribution of QoI estimated with importance sampling
    mean_importance_sample, var_importance_sample = get_mean_var(
        qoi_estimator_importance_sample, torch.tensor(results_importance_sample[idx])
    )
    qoi_dist_importance_sample = Normal(mean_importance_sample, var_importance_sample**0.5)
    _ = population_estimators.plot_dist(
        qoi_dist_importance_sample, ax=ax, c="tab:red", label="QOI estimate (importance sampling)"
    )

    # Create distribution of QoI estimated without importance sampling
    mean_whole_dataset, var_whole_dataset = get_mean_var(
        qoi_estimator_whole_dataset, torch.tensor(results_whole_dataset[idx])
    )
    qoi_dist_whole_dataset = Normal(mean_whole_dataset, var_whole_dataset**0.5)
    _ = population_estimators.plot_dist(qoi_dist_whole_dataset, ax=ax, c="tab:blue", label="QOI estimate")

    ax.axvline(brut_force_qoi, c="orange", label="Brute force results")

    ax.set_title(f"QoI estimate with {n_points} training points")
    ax.set_xlabel("Response")
    ax.set_ylabel("Density")
    ax.legend()

# %%
