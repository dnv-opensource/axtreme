# %% [markdown]
# # Importance sampling tutorial
#
# In `axtreme`, importance sampling is used to improve the estimation of the quantity of interest (QoI) when the
# extreme response is concentrated in only part of the input space.
#
# Traditional sampling strategies like Sobol sampling aim to cover the entire search space uniformly. This works
# well when the phenomenon of interest is spread out. However, when the extreme response we're modeling
# occurs in a small, specific region, uniform sampling becomes inefficient. In such cases, importance
# sampling allows us to focus more training points in the relevant region, improving accuracy and reducing
# uncertainty in the QoI estimate.
#
# In practical terms, this means that we may need fewer expensive model evaluations to get a good (i.e. with low
# uncertainty) QoI estimate.
#
# ## Theory
# For a detailed explanation of the theory behind importance sampling as implemented in `axtreme`, see:
#
# Winter, S. et al. (2025) "Efficient Long-Term Structural Reliability Estimation with Non-Gaussian Stochastic Models:
# A Design of Experiments Approach". arXiv. https://doi.org/10.48550/arXiv.2503.01566.
#
# This tutorial focuses on how to use importance sampling in `axtreme`, rather than explaining the underlying
# mathematics in depth. To get an understanding of how `axtreme` works check out basic_example.py.
#
# `axtreme` currently supports importance sampling when the environment distribution is known. If the distribution
# is not known, it can be estimated, for example, via Kernel Density Estimation (KDE)—though that is outside the
# scope of this tutorial.

# ## The setup for this tutorial
#
# To clearly demonstrate the benefits of importance sampling in `axtreme`, we use an example where:
#
# 1. The extreme response is concentrated in a small region.
# 2. The environment data is concentrated in one region and only little environment data is covering the region with
#   the extreme response.

# %%
import sys
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.stats as st
import torch
from ax import (
    Experiment,
    SearchSpace,
)
from ax.core import ParameterType, RangeParameter
from ax.modelbridge.registry import Models
from matplotlib.axes import Axes
from scipy.stats import gumbel_r
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

# TODO (ak:25-08-06): change path when file is moved to src
from examples.crest_heights_north_sea.importance_sampling import importance_sampling_distribution_uniform_region
from examples.tutorials.importance_sampling.problem.brute_force import collect_or_calculate_results
from examples.tutorials.importance_sampling.problem.env_data import calculate_environment_distribution, collect_data
from examples.tutorials.importance_sampling.problem.simulator import DummySimulatorSeeded

# %% [markdown]
# ## Environment data

# The environment data is given by a multivariate normal distribution with mean $\mu=[0.1, 0.1]$
# and covariance $\Sigma=[[0.2, 0], [0, 0.2]]$. The data is limited to values between 0 and 1
# to remove outliers.

env_data = collect_data().to_numpy()

fig = histogram_surface3d(env_data, density=False)
_ = fig.update_layout(title_text="Histogram of environment samples")
_ = fig.update_layout(scene_aspectmode="cube")
fig.show()


# %% [markdown]
# ## Brute force solution of the extreme response
# The simulator used for this tutorial is Multivariate normal distribution with mean $\mu=[1,1]$ and covariance
# $\Sigma=[[0.03, 0], [0, 0.03]]$.
#
# To have a value to judge the accuracy of the QoI estimate derived by axtreme the true solution is
# estimated using a brute force approach. This is generally not possible in a real use case but allows
# us to better demonstrate the effect of importance sampling in this tutorial.

# %%
# Number of samples in a period
N_ENV_SAMPLES_PER_PERIOD = 1000

# Calculate ERD samples and their location using a brute force approach
precalced_erd_samples, precalced_erd_x = collect_or_calculate_results(N_ENV_SAMPLES_PER_PERIOD, 400_000)
brute_force_qoi_estimate = np.median(precalced_erd_samples)
print(f"Brute force estimate of our QOI is {brute_force_qoi_estimate}")

# %% [markdown]
# ### Brute force extreme response locations
# This shows that the extreme responses (ERs) are mainly clustered together in one
# region and there is not much env data covering the region where most ER are observed.

fig_combined = go.Figure()

# Add environment data traces
fig_env = histogram_surface3d(
    env_data, surface3d_kwargs={"colorscale": "Blues", "name": "Environment Data", "showscale": False}
)
_ = fig_combined.add_trace(fig_env.data[0])

# Add extreme response location traces
fig_extreme = histogram_surface3d(
    precalced_erd_x.numpy(),
    surface3d_kwargs={"colorscale": "Reds", "name": "Extreme Response Locations", "showscale": False},
)
_ = fig_combined.add_trace(fig_extreme.data[0])

_ = fig_combined.update_layout(
    title_text="Density of the environment data (blue)"
    "<br>vs the density of the location of the extreme responses (red)",
    scene={"zaxis": {"title": "Density of ER location"}},
)
fig_combined.show()


# %% [markdown]
# ## Importance samples with an unknown importance sample distribution
# The importance samples and weights are generated by sampling uniformly from a defined region of interest. If the
# importance sample distribution is known the samples and weights can be generated using the function
# `importance_sampling_from_distribution`.

# To create the importance samples and weights using a uniform region the following parameters need to be set:
# 1. env_distribution_pdf: The pdf function of the real environment. This can either be a known function or
#   for example be estimated based on the environment data using KDE (kernel density estimation). For this example it is
#   known and defined in the function calculate_environment_distribution
# 2. region: The bounds of the region to generate samples from.
# 3. threshold: Environment regions with pdf values less than this threshold will not be included in the importance
#   samples.
# 4. num_samples_total: Total number of samples to draw uniformly before filtering. The actual number of returned
#   samples may be smaller depending on how many pass the threshold filter.

# We now want to explore how these parameters can be chosen and their impact.

# ### How to choose the region
# The simulator used for this tutorial is Multivariate normal distribution with mean $\mu=[1,1]$ and covariance
# $\Sigma=[[0.03, 0], [0, 0.03]]$. Therefore, we expect the extreme responses to be centered mainly around
# $(x_1,x_2)=(1,1)$. This can be verified by looking at the figure of the brute force estimate of the extreme
# response locations.

# %%
# We are showing the effect of the region size for three different ones
regions = {
    "too large": torch.tensor([[0.0, 0.0], [10.0, 10.0]]),
    "too small": torch.tensor([[0.0, 0.0], [1.0, 1.0]]),
    "good fit": torch.tensor([[0.0, 0.0], [2.0, 2.0]]),
}

importance_samples_all_regions = []
importance_weights_all_regions = []

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
axes = axes.ravel()

for ax, title, region in zip(axes, regions.keys(), regions.values(), strict=False):
    importance_samples, importance_weights = importance_sampling_distribution_uniform_region(
        env_distribution_pdf=calculate_environment_distribution,
        region=region,
        threshold=1e-3,
        num_samples_total=10_000,
    )
    importance_samples_all_regions.append(importance_samples)
    importance_weights_all_regions.append(importance_weights)

    _ = ax.scatter(importance_samples[:, 0], importance_samples[:, 1], s=2, c="tab:blue", label="Importance samples")
    _ = ax.scatter(precalced_erd_x[:, 0], precalced_erd_x[:, 1], s=2, c="tab:orange", label="ER locations")
    _ = ax.set_title(f"Region: {title}")
    _ = ax.set_xlabel("x1")
    _ = ax.set_ylabel("x2")
    ax.legend()

fig.tight_layout()

# %% [markdown]
# From the first plot we can clearly see that when the region is chosen to be too large only very few samples cover the
# important region. The second plot shows that if the region is chosen too small important areas of the environment
# space are missed. This can lead to a wrong estimation of the QoI. In the third plot all extreme responses are covered
# by the importance samples while still maintaining a significant amount of coverage. In a real use case the ER
# locations are often not known. In that case domain knowledge can be used to choose the region. If in doubt
# a larger region with a higher number of samples is the safer choice than a too small region. This will however
# be more computationally expensive.


# %% [markdown]
# ## How to choose the threshold
# In the previous figure we showed that the region $x_1, x_2 \in [0.5,1.5]$ is good fit for this example. Now we want
# to show the impact of the threshold for this region. The threshold basically defines which part of the region shall be
# ignored based on the probability of the environment data being lower than this threshold. More details are given in
# the paper by Winter et.al.
#
# The following figure shows the density of the environment data estimated based on the available env_data.
xmin, xmax, ymin, ymax = 0, 2, 0, 2
xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
positions = np.vstack([xx.ravel(), yy.ravel()])

kde = st.gaussian_kde(env_data.T)(positions).reshape(xx.shape)

fig, ax = plt.subplots()
_ = ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax), xlabel="x1", ylabel="x2")

levels = [0, 0.00001, 0.0001, 0.001, 0.01, 0.1, 0.2, 1, 2, 3]
cf = ax.contourf(xx, yy, kde, levels=levels, cmap="Blues")
c = ax.contour(xx, yy, kde, levels=levels, colors="k")
_ = ax.clabel(c, inline=1, fontsize=10, fmt="%.1e")

_ = ax.scatter(precalced_erd_x[:, 0], precalced_erd_x[:, 1], s=2, c="tab:orange", label="ER locations")
_ = ax.legend()
_ = ax.set_title("PDF of environment data")

# %% [markdown]
# This plot shows that the extreme responses are located in a region where the pdf of the environment data is between
# 1e-2 and 2e-1. Therefore, we can assume that a threshold of 1e-3 should include all of the extreme response without
# including too much unwanted data.

# We show the impact of choosing the threshold too low or too high and compare the results to the ones with the
# threshold given by the pdf.

thresholds = {"low": 1e-5, "high": 1e-1, "good fit": 1e-3}

importance_samples_all_thresholds = []
importance_weights_all_thresholds = []

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
axes = axes.ravel()

for ax, title, threshold in zip(axes, thresholds.keys(), thresholds.values(), strict=False):
    importance_samples, importance_weights = importance_sampling_distribution_uniform_region(
        env_distribution_pdf=calculate_environment_distribution,
        region=regions["good fit"],
        threshold=threshold,
        num_samples_total=10_000,
    )
    importance_samples_all_thresholds.append(importance_samples)
    importance_weights_all_thresholds.append(importance_weights)

    _ = ax.scatter(importance_samples[:, 0], importance_samples[:, 1], s=2, c="tab:blue", label="Importance samples")
    _ = ax.scatter(precalced_erd_x[:, 0], precalced_erd_x[:, 1], s=2, c="tab:orange", label="ER locations")
    _ = ax.set_title(f"Threshold: {title}")
    _ = ax.set_xlabel("x1")
    _ = ax.set_ylabel("x2")
    _ = ax.legend()

fig.tight_layout()

# %% [markdown]
# As can be seen in the first plot if the threshold is very low the whole region is sampled which means that the pdf
# values have little to no impact. As a result a larger sampling area than necessary is chosen thereby reducing the
# effectiveness of importance sampling compared to Sobol sampling. By choosing the threshold too high important parts
# important areas of the environment space are missed (2nd plot). This can lead to a wrong estimation of the QoI.
# In the third plot all extreme responses are covered by the samples. Thereby we conclude that a threshold of 1e-3 is a
# good fit for this example.

best_importance_samples = importance_samples_all_thresholds[2]
best_importance_weights = importance_weights_all_thresholds[2]

# %% [markdown]
# The following plots shows the importance samples and colored according to their corresponding weight for the best
# threshold and region chosen by the analysis above.

sc = plt.scatter(
    best_importance_samples[:, 0],
    best_importance_samples[:, 1],
    s=1,
    c=best_importance_weights,
    label="Importance samples",
)
_ = plt.colorbar(sc, label="Importance weights")
_ = plt.title("Importance samples")
_ = plt.xlabel("x1")
_ = plt.ylabel("x2")

# %% [markdown]
# ## Using the importance samples in axtreme
# Now that we have the importance samples and weights we can use them in `axtreme` to estimate the QoI.
# The importance samples and weights are wrapped in an `ImportanceAddedWrapper`. This wrapper allows us to use the
# importance samples and weights in the QoI estimation.

importance_dataset = ImportanceAddedWrapper(
    MinimalDataset(best_importance_samples), MinimalDataset(best_importance_weights)
)

# Save the importance samples and weights in files
file_path = Path(__file__).parent.parent / "examples/tutorials/importance_sampling/problem/data"
torch.save(best_importance_samples, f"{file_path}/importance_samples.pt")
torch.save(best_importance_weights, f"{file_path}/importance_weights.pt")

# %% [markdown]
# ### Setting up parameters/ functions required for axtreme

search_space = SearchSpace(
    parameters=[
        RangeParameter(name="x1", parameter_type=ParameterType.FLOAT, lower=0, upper=1),
        RangeParameter(name="x2", parameter_type=ParameterType.FLOAT, lower=0, upper=1),
    ]
)

dist = gumbel_r

# Pick a number of GP simulations per point
N_SIMULATIONS_PER_POINT = 30

# Use seeded simulator to avoid introducing additional uncertainty through a stochastic simulator
sim = DummySimulatorSeeded()


def make_exp() -> Experiment:
    """Helper to ensure we always create an experiment with the same settings (so results are comparable)."""
    return make_experiment(sim, search_space, dist, n_simulations_per_point=N_SIMULATIONS_PER_POINT)


exp = make_exp()


# Create a single GP to run the QoI with 100 training points
add_sobol_points_to_experiment(exp, n_iter=100, seed=8)

botorch_model_bridge = Models.BOTORCH_MODULAR(
    experiment=exp,
    data=exp.fetch_data(),
)

input_transform, outcome_transform = transforms.ax_to_botorch_transform_input_output(
    transforms=list(botorch_model_bridge.transforms.values()), outcome_names=botorch_model_bridge.outcomes
)

# %% [markdown]
# ### Run the QoI estimation with and without importance sampling

qoi_jobs = []
datasets = {"full": env_data, "importance_sample": importance_dataset}
for dataset_name, dataset in datasets.items():
    # TODO (ak:25-08-07): verify that this is the correct interpretation if num_samples

    # To calculate the QoI not the whole environment data is used but only a subset of it. The size of this subset is
    # given by `dataset_size`. We expect the QoI estimate to be mor accurate and have less variability for a larger
    # dataset size.
    for dataset_size in [100, 200, 1000, 8000]:
        # We run the QoI estimation 200 times for the same parameters to give a clear picture of the uncertainty
        # of the QoI estimate.
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

jobs_output_file = None
qoi_results = [job(output_file=jobs_output_file) for job in qoi_jobs]

df_jobs = pd.json_normalize([item.to_dict() for item in qoi_results], max_level=1)
df_jobs.columns = df_jobs.columns.str.removeprefix("tags.")
df_jobs.head()


# %% [markdown]
# ### Plotting the results
# The following plots show the QoI estimate for the different dataset sizes with and without importance sampling.
#
#  The following functions are helpers to simplify plotting
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
        partial(plot_qoi_as_normal, brute_force=float(brute_force_qoi_estimate)),
        partial(plot_best_guess, brute_force=float(brute_force_qoi_estimate)),
        partial(plot_col_histogram, col_name="var"),
    ],
)

# %%
