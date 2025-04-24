"""Evaluate the convergence of the QOI for different training datasets."""

# %%
import matplotlib.pyplot as plt
import numpy as np
import simulator  # type: ignore[import]
import torch
from ax import (
    Experiment,
    SearchSpace,
)
from ax.core import ParameterType, RangeParameter
from ax.modelbridge.registry import Models
from problem import brut_force_qoi, period_length  # type: ignore[import-not-found]
from scipy.stats import gumbel_r
from torch.distributions import Normal
from torch.utils.data import DataLoader

from axtreme.data import FixedRandomSampler
from axtreme.data.dataset import MinimalDataset
from axtreme.experiment import add_sobol_points_to_experiment, make_experiment
from axtreme.qoi import MarginalCDFExtrapolation
from axtreme.qoi.qoi_estimator import QoIEstimator
from axtreme.sampling.ut_sampler import UTSampler
from axtreme.utils import population_estimators, transforms

# %%
# Set up Qoi estimator
n_env_samples = 1_000

## Load environment data
data = np.load("data/long_term_distribution.npy")
dataset = MinimalDataset(data)

## A random dataloader give different env samples for each instance
sampler = FixedRandomSampler(dataset, num_samples=n_env_samples, seed=10, replacement=True)
dataloader = DataLoader(dataset, sampler=sampler, batch_size=256)

posterior_sampler = UTSampler()

qoi_estimator = MarginalCDFExtrapolation(
    env_iterable=dataloader,
    period_len=period_length,
    quantile=torch.tensor(0.5),
    quantile_accuracy=torch.tensor(0.01),
    posterior_sampler=posterior_sampler,
)

# %%
# Set up experiment
## Define distribution
DIST = gumbel_r

## Define simulator
sim = simulator.MaxCrestHeightSimulator()

## Pick the search space over which to create a surrogate
# For now this is based on the min and max of the env data/long_term_distribution_100_years.npy
SEARCH_SPACE = SearchSpace(
    parameters=[
        RangeParameter(name="Hs", parameter_type=ParameterType.FLOAT, lower=7.5, upper=20),
        RangeParameter(name="Tp", parameter_type=ParameterType.FLOAT, lower=7.5, upper=20),
    ],
)


# %%
def make_exp() -> Experiment:
    """Convenience function returns a fresh Experiment of this problem."""
    return make_experiment(sim, SEARCH_SPACE, DIST, n_simulations_per_point=n_env_samples)


# %%
n_training_points = [30, 50]  # , 128 , 512]
results = []

for points in n_training_points:
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
    # reseed the dataloader each time so the dame dataset is used.
    results.append(qoi_estimator(model))


# %%
def get_mean_var(estimator: QoIEstimator, estimates: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """TODO: clean this up or delete.

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


# %%
fig, axes = plt.subplots(nrows=len(n_training_points), sharex=True, figsize=(6, 6 * len(n_training_points)))

for ax, estimate, n_points in zip(axes, results, n_training_points, strict=True):
    mean, var = get_mean_var(qoi_estimator, torch.tensor(estimate))
    qoi_dist = Normal(mean, var**0.5)
    _ = population_estimators.plot_dist(qoi_dist, ax=ax, c="tab:blue", label="QOI estimate")

    ax.axvline(brut_force_qoi, c="orange", label="Brute force results")

    ax.set_title(f"QoI estimate with {n_points} training points")
    ax.set_xlabel("Response")
    ax.set_ylabel("Density")
    ax.legend()

# %%
