"""This explores the bias and variance of a QoI estimator with different hyperparameters.

A QoI estimator should be chosen that has sufficiently small bias and variance, as typically a single instance of
the estimator is used for later DOE steps. As the DOE runs the QoI estimator many time, it is beneficial if it is fast.

This file does the following:
- define a model
- create many instances of 4 different QoI estimators. Specifically:
    - random env sample - 1000 points
    - random env sample - 8000 points
    - importance sample - 1000 points
    - importance sample - 8000 points
- Runs the jobs
- Plots the result in order to understand the bias and variance
"""

# %%
from functools import partial

import matplotlib.pyplot as plt
import pandas as pd
import torch
from ax.modelbridge.registry import Models
from matplotlib.axes import Axes
from problem import brute_force_qoi, importance_dataset, make_exp, period_length
from problem import dataset as mc_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from axtreme.data import FixedRandomSampler
from axtreme.eval.qoi_helpers import plot_col_histogram, plot_groups
from axtreme.eval.qoi_job import QoIJob
from axtreme.experiment import add_sobol_points_to_experiment
from axtreme.qoi import MarginalCDFExtrapolation
from axtreme.sampling.ut_sampler import UTSampler
from axtreme.utils import population_estimators, transforms

# %% create a single GP to run the QoI with
exp = make_exp()
add_sobol_points_to_experiment(exp, n_iter=100, seed=8)
botorch_model_bridge = Models.BOTORCH_MODULAR(
    experiment=exp,
    data=exp.fetch_data(),
)

# We need to collect the transforms used to the model gives result in the problem/outcome space.
input_transform, outcome_transform = transforms.ax_to_botorch_transform_input_output(
    transforms=list(botorch_model_bridge.transforms.values()), outcome_names=botorch_model_bridge.outcomes
)

# %% Create qois with different variants of dataset and samples
# NOTE: the could be simplified by just using a random sampler, but this is a bit more obvious what its doing.
qoi_jobs = []
datasets = {"mc": mc_dataset, "importance_sample": importance_dataset}
for dataset_name, dataset in datasets.items():
    for dataset_size in [1_000, 8_000]:
        for i in range(20):
            sampler = FixedRandomSampler(dataset, num_samples=dataset_size, seed=i, replacement=True)
            dataloader = DataLoader(dataset, sampler=sampler, batch_size=256)

            posterior_sampler = UTSampler()

            qoi_estimator = MarginalCDFExtrapolation(
                env_iterable=dataloader,
                period_len=period_length,
                quantile=torch.exp(torch.tensor(-1)),
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


# %% Run the jobs
# If a path is provided here the jobs will be saved to file. They can then be loaded later for analysis
jobs_output_file = None
qoi_results = [job(output_file=jobs_output_file) for job in tqdm(qoi_jobs)]

# %% create a a Dataframe from the results. This can also be done by ready the output file if one was created.
df_jobs = pd.json_normalize([item.to_dict() for item in qoi_results], max_level=1)
df_jobs.columns = df_jobs.columns.str.removeprefix("tags.")
df_jobs.head()  # pyright: ignore[reportUnusedCallResult]


# %%
"""Explore and plot the results.

The following is done:
- Plot a single subset of the data to demonstrate the plotting functionality
- Plot each group of data.
- Produce additional plots for the importance sampling results.
- Compare the runtime of the different QoI estimators.
"""


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


# %% Demonstrate plotting on subset (this can be any subset, but it makes most sense if they come from a single group)
df_subset = df_jobs[:20]
_, axes = plt.subplots(1, 3, figsize=(15, 5))
plot_qoi_as_normal(df_subset, axes[0], brute_force=brute_force_qoi)
plot_best_guess(df_subset, axes[1], brute_force=brute_force_qoi)
plot_col_histogram(df_subset, axes[2], col_name="var", brute_force=None)


# %% Plot all of the groups
df_grouped = df_jobs.groupby(["dataset_name", "dataset_size"])
_ = plot_groups(
    df_grouped,
    plotting_funcs=[
        partial(plot_qoi_as_normal, brute_force=brute_force_qoi),
        partial(plot_best_guess, brute_force=brute_force_qoi),
        partial(plot_col_histogram, col_name="var"),
    ],
)

# %% Additional plotting for importance ssamples

df_grouped_is = df_jobs[df_jobs["dataset_name"] == "importance_sample"].groupby(["dataset_size"])
_ = plot_groups(
    df_grouped_is,
    plotting_funcs=[
        partial(plot_qoi_as_normal, brute_force=brute_force_qoi),
        partial(plot_best_guess, brute_force=brute_force_qoi),
        partial(plot_col_histogram, col_name="var"),
    ],
)
# %% General exploration of Runtime
df_grouped["metadata.runtime_sec"].agg(["mean", "var"])  # pyright: ignore[reportUnusedCallResult]
# %%
"""
On the basis of this analysis we determine that the QOI estimator with 8000 importance samples has suitable bias
and variance for our needs.
"""
