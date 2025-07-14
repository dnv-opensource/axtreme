"""
Steps needed to be done:
1. define simulator and environment such that
- env data does not cover the whole search space
- extreme response is concentrated in one small region where there is little coverage of the env data
2. calculate brute force QoI
3. calculate QoI without importance sampling
4. calculate QoI with importance sampling

Wanted Outcome:
-   Qoi with importance sampling needs less training points to get close to brute force solution than without importance
    sampling
"""

# %%
import sys
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from ax import (
    Experiment,
    SearchSpace,
)
from ax.core import ParameterType, RangeParameter
from ax.modelbridge.registry import Models
from matplotlib.axes import Axes
from scipy.stats import gumbel_r
from torch.distributions import MultivariateNormal
from torch.utils.data import DataLoader

from axtreme.data import FixedRandomSampler, ImportanceAddedWrapper, MinimalDataset
from axtreme.eval.qoi_helpers import plot_col_histogram, plot_groups
from axtreme.eval.qoi_job import QoIJob
from axtreme.experiment import add_sobol_points_to_experiment, make_experiment
from axtreme.plotting.histogram3d import histogram_surface3d
from axtreme.qoi import MarginalCDFExtrapolation
from axtreme.sampling.ut_sampler import UTSampler
from axtreme.utils import population_estimators, transforms

torch.set_default_dtype(torch.float64)

root_dir = Path("../")
sys.path.append(str(root_dir))
from examples.basic_example_usecase.problem.brute_force import collect_or_calculate_results
from examples.basic_example_usecase.problem.env_data import collect_data
from examples.basic_example_usecase.problem.simulator import (
    DummySimulatorSeeded,
)
from examples.crest_heights_north_sea.importance_sampling import importance_sampling_distribution_uniform_region

# %%
# Load environment data
env_data = collect_data().to_numpy()

fig = histogram_surface3d(env_data)
_ = fig.update_layout(title_text="Environment distribution estimate from samples")
_ = fig.update_layout(scene_aspectmode="cube")
fig.show()

# %%
# Use seeded simulator to avoid introducing additional uncertainty through a stochastic simulator
sim = DummySimulatorSeeded()

# define the time span
N_ENV_SAMPLES_PER_PERIOD = 1000

# %%
# Brute force solution

precalced_erd_samples, precalced_erd_x = collect_or_calculate_results(N_ENV_SAMPLES_PER_PERIOD, 200_000)
brute_force_qoi_estimate = np.median(precalced_erd_samples)
print(f"Brute force estimate of our QOI is {brute_force_qoi_estimate}")

# %%
# Brute Force Extreme Response Locations
# Create combined histogram of environment data and extreme responses
fig_combined = go.Figure()

# Add environment data traces
fig_env = histogram_surface3d(
    env_data, surface3d_kwargs={"colorscale": "Blues", "name": "Environment Data", "showscale": False}
)
_ = fig_combined.add_trace(fig_env.data[0])

# Add extreme responses traces
fig_extreme = histogram_surface3d(
    precalced_erd_x.numpy(), surface3d_kwargs={"colorscale": "Reds", "name": "Extreme Responses", "showscale": False}
)
_ = fig_combined.add_trace(fig_extreme.data[0])

_ = fig_combined.update_layout(title_text="Environment Data(blue) vs Extreme Responses(red)")
fig_combined.show()

# This shows exactly what we wanted the extreme responses are mainly clustered together in one
# region (need to see if having a second location is an issue) and there is not much env data
# covering the region where most ER are observed.

# %%
# Create importance samples


# Define env data distribution
# TODO (ak:25-07-09): Move to env_data.py
def calculate_environment_distribution(samples: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.1, 0.1])
    cov = torch.tensor([[0.2, 0], [0, 0.2]])
    distribution = MultivariateNormal(mean, covariance_matrix=cov)

    # log_prob returns the log of the pdf evaluated at a value. Pytorch does not provide the pdf
    # directly. Therefore, we need to use the exponential of the log pdf.
    return torch.exp(distribution.log_prob(samples))


importance_samples, importance_weights = importance_sampling_distribution_uniform_region(
    env_distribution_pdf=calculate_environment_distribution,
    region=torch.tensor([[0.0, 0.0], [2.0, 2.0]]),
    threshold=1e-3,
    num_samples_total=10_000,
)
importance_dataset = ImportanceAddedWrapper(MinimalDataset(importance_samples), MinimalDataset(importance_weights))

# %%
torch.save(importance_samples, "importance_samples.pt")
torch.save(importance_weights, "importance_weights.pt")

# %%
sc = plt.scatter(
    importance_samples[:, 0], importance_samples[:, 1], s=1, c=importance_weights, label="Importance samples"
)
_ = plt.colorbar(sc, label="Importance weights")
_ = plt.title("Importance samples")
_ = plt.legend()

# %%
_ = plt.scatter(importance_samples[:, 0], importance_samples[:, 1], s=2, c="tab:blue", label="Importance samples")
_ = plt.scatter(precalced_erd_x[:, 0], precalced_erd_x[:, 1], s=2, c="tab:orange", label="Brute force samples")
_ = plt.title("Importance samples vs brute force xmax samples")
_ = plt.legend()
# %%
# Set up things needed for axtreme
search_space = SearchSpace(
    parameters=[
        RangeParameter(name="x1", parameter_type=ParameterType.FLOAT, lower=0, upper=2),
        RangeParameter(name="x2", parameter_type=ParameterType.FLOAT, lower=0, upper=2),
    ]
)

dist = gumbel_r

# Pick a number of simulations per point. Higher values = less uncertainty in the GP fit.
N_SIMULATIONS_PER_POINT = 30


def make_exp() -> Experiment:
    """Helper to ensure we always create an experiment with the same settings (so results are comparable)."""
    return make_experiment(sim, search_space, dist, n_simulations_per_point=N_SIMULATIONS_PER_POINT)


exp = make_exp()


# %% create a single GP to run the QoI with
add_sobol_points_to_experiment(exp, n_iter=100, seed=8)

botorch_model_bridge = Models.BOTORCH_MODULAR(
    experiment=exp,
    data=exp.fetch_data(),
)

input_transform, outcome_transform = transforms.ax_to_botorch_transform_input_output(
    transforms=list(botorch_model_bridge.transforms.values()), outcome_names=botorch_model_bridge.outcomes
)

# %% Create jobs with with and without importance sampling
qoi_jobs = []
datasets = {"full": env_data, "importance_sample": importance_dataset}
for dataset_name, dataset in datasets.items():
    for dataset_size in [100, 200, 1000, 5000, 10000]:
        for i in range(200):
            sampler = FixedRandomSampler(dataset, num_samples=dataset_size, seed=i, replacement=True)
            dataloader = DataLoader(dataset, sampler=sampler, batch_size=100)

            posterior_sampler = UTSampler()

            qoi_estimator = MarginalCDFExtrapolation(
                env_iterable=dataloader,
                period_len=N_ENV_SAMPLES_PER_PERIOD,
                quantile=torch.tensor(0.5),
                quantile_accuracy=torch.tensor(0.01),
                posterior_sampler=posterior_sampler,
            )

            qoi_estimator.input_transform = input_transform
            qoi_estimator.outcome_transform = outcome_transform

            qoi_jobs.append(
                QoIJob(
                    name=f"qoi_{dataset_name}_{dataset_size}_{i}",
                    qoi=qoi_estimator,
                    model=botorch_model_bridge.model.surrogate.model,
                    tags={
                        "dataset_name": dataset_name,
                        "dataset_size": dataset_size,
                    },
                )
            )

# %% create a a Dataframe from the results. This can also be done by ready the output file if one was created.
jobs_output_file = None
qoi_results = [job(output_file=jobs_output_file) for job in qoi_jobs]

df_jobs = pd.json_normalize([item.to_dict() for item in qoi_results], max_level=1)
df_jobs.columns = df_jobs.columns.str.removeprefix("tags.")
df_jobs.head()


# %% Custom plotting helpers
def plot_best_guess(df: pd.DataFrame, ax: Axes, brute_force: float | None = None) -> None:
    """Plots `plot_col_histogram` with the mean and standard error of this estimate added.

    Note: This function follows the interface in axtreme.eval.qoi_plots
    """
    plot_col_histogram(df, ax, brute_force=brute_force)

    # Add the se plot
    ax_twin = ax.twinx()
    samples = torch.tensor(df.loc[:, "mean"].to_numpy())
    sample_mean_se_dist = population_estimators.sample_mean_se(samples)
    _ = population_estimators.plot_dist(sample_mean_se_dist, ax=ax_twin, c="red", label="dist mean 99.7% conf interval")
    _ = ax_twin.legend()


def plot_qoi_as_normal(
    df: pd.DataFrame,
    ax: Axes,
    n_plots: int = 10,
    mean_col: str = "mean",
    var_col: str = "var",
    brute_force: float | None = None,
) -> None:
    """Quick and dirty function for plotting the QoI output distribution (here assumed normal).

    Note: this is particularly suitable when UTtransform is used as the posterior samples.
    For other posterior samplers, plot_histogram is more expressive.

    Args:
        df: A dataframe.
        ax: The axis to plot on.
        n_plots: The number of rows from the df to plot.
        mean_col: the name of the column containing the mean of the QoI estimate (should be of type float).
        var_col: the name of the column containing the variance of the QoI estimate (should be of type float).
        brute_force: Represents the true value (e.g mean). Plots a vertical line if provided.
    """
    mean = df.loc[:, mean_col].to_numpy()[:n_plots]
    var = df.loc[:, var_col].to_numpy()[:n_plots]

    lower_bound = (mean - 4 * var**0.5).min()
    upper_bound = (mean + 4 * var**0.5).max()
    x = torch.linspace(lower_bound, upper_bound, 1000)

    for samples_i in zip(mean, var, strict=True):
        _ = ax.plot(x, torch.distributions.Normal(samples_i[0], samples_i[1] ** 0.5).log_prob(x).exp(), alpha=0.3)

    _ = ax.set_xlabel("QOI estimate")
    _ = ax.set_title("QOI estimator distributions (+- 4 stdev)")

    if brute_force:
        _ = ax.axvline(brute_force, c="black", label=f"Brute force ({brute_force:.2f})")

    _ = ax.legend()


# %% Plot all of the groups
df_grouped = df_jobs.groupby(["dataset_name", "dataset_size"])
_ = plot_groups(
    df_grouped,
    plotting_funcs=[
        partial(plot_qoi_as_normal, brute_force=brute_force_qoi_estimate),
        partial(plot_best_guess, brute_force=brute_force_qoi_estimate),
        partial(plot_col_histogram, col_name="var"),
    ],
)

# %%
