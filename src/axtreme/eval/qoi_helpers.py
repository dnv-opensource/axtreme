"""Plotting helper tailored for analysingQoIJobResults."""

import copy
from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from botorch.models import PosteriorMeanModel, SingleTaskGP
from botorch.sampling.index_sampler import IndexSampler
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pandas.core.groupby.generic import DataFrameGroupBy

from axtreme.qoi.gp_bruteforce import GPBruteForce


def plot_distribution(
    df: pd.DataFrame,
    ax: Axes,
    n_hists: int = 3,
    col_name: str = "samples",
    brute_force: float | None = None,
) -> None:
    """Helperfor plotting histograms (on the given ax) of dataframe cells containing lists.

    Designed for use with the 'samples' column of a QoiJobResults dataframe.

    Args:
        df: A dataframe.
        ax: The axis to plot on.
        n_hists: The number of cells of column `col_name` to plot.
        col_name: The column of the df containing lists.
        brute_force: Represents the true value (e.g mean). Plots a vertical line if provided.
    """
    samples_list = df.loc[:, col_name].to_numpy()
    samples_list = samples_list[:n_hists]

    for samples_i in samples_list:
        _ = ax.hist(samples_i, density=True, alpha=0.3, bins=len(samples_i) // 5 + 1)
    _ = ax.set_xlabel("QOI value")
    _ = ax.set_title("QOi estimator distibutions")

    if brute_force:
        _ = ax.axvline(brute_force, c="black", label=f"Brute force ({brute_force:.2f})")

    _ = ax.legend()


def plot_col_histogram(df: pd.DataFrame, ax: Axes, col_name: str = "mean", brute_force: float | None = None) -> None:
    """Helper which creates a histogram (on the given ax) based on a column of the df.

    Designed for use with the 'mean' or 'var' column of a QoiJobResults dataframe.

    Args:
        df: A dataframe.
        ax: The axis to plot on.
        col_name: The column of the df containing lists.
        brute_force: Represents the true value (e.g mean). Plots a vertical line if provided.
    """
    values = df.loc[:, col_name].to_numpy()
    _ = ax.hist(values, label=f"{col_name} of qoi runs", density=True, bins=30)
    title_str = (
        f"{col_name} of each qoi run\n"
        f"mean of dist {values.mean():.3f}."
        f" std of dist {values.std():.3f},"
        # Protect against divide by error error
        f"C.o.V {values.std()/ values.mean() if values.mean() > 1e-2 else np.nan:.3f}"  # noqa: PLR2004
    )
    _ = ax.set_title(title_str)
    _ = ax.set_ylabel("density")
    _ = ax.set_xlabel("QOI value")

    if brute_force:
        _ = ax.axvline(brute_force, c="black", label=f"Brute force ({brute_force:.2f})")

    _ = ax.legend()


def plot_groups(
    # Error if try to parmaterise this as a generic
    df_grouped: DataFrameGroupBy,  ## pyright: ignore[reportUnknownParameterType,reportMissingTypeArgument]
    plotting_funcs: list[Callable[[pd.DataFrame, Axes], None]],
) -> Figure:
    """Takes a grouped dataframe, and generates a row of plots for each group, using plotting_funcs.

    Args:
        df_grouped: The groupby object to plot
        plotting_funcs: list of plots to be generated for each group. See plot_col_histogram for an example of a
          plotting function.
    """
    group_names = df_grouped.groups
    n_groups = len(group_names)
    n_cols = len(plotting_funcs)
    fig, axes = plt.subplots(nrows=n_groups, ncols=n_cols, figsize=(6 * n_cols, 4 * n_groups), sharex="col")

    if n_groups == 1:
        # reshape so can be handled in the same manner as 2d
        axes = axes.reshape(1, n_cols)

    for i, (_, group_data) in enumerate(df_grouped):
        row_axes = axes[i]
        for plot_func, ax in zip(plotting_funcs, row_axes, strict=True):
            plot_func(group_data, ax)

    # Add group labels to each row
    for ax, row in zip(axes[:, 0], group_names, strict=True):
        ax.annotate(
            row,
            xy=(0, 0.5),
            xytext=(-ax.yaxis.labelpad - 5, 0),
            xycoords=ax.yaxis.label,
            textcoords="offset points",
            size="large",
            ha="right",
            va="center",
        )

    fig.tight_layout()
    # tight_layout doesn't take these labels into account. Manually tweak.
    fig.subplots_adjust(left=0.15)

    return fig


def qoi_ignoring_gp_uncertainty(qoi: GPBruteForce, model: SingleTaskGP) -> torch.Tensor:
    """Helper to run a QoI with a model, ignoring uncertainty in the model (e.g using the posterior mean).

    Args:
        qoi: The QoI estimator to use
        model: The model to use

    Return:
        the estimates made by the QoI using only the posterior mean of the model.
    """
    # copy the QoI as we will need to change some internal settings
    new_qoi = copy.deepcopy(qoi)

    # make a deterministic model
    det_model = PosteriorMeanModel(model)

    # We need to change the sampler to be compatible with a deterministic model

    if not hasattr(new_qoi.posterior_sampler, "sample_shape"):
        msg = (
            "This function only works with samplers that have a sample_shape attribute."
            " e.g Like those inheriting from botorch MCSampler."
        )
        raise NotImplementedError(msg)

    sample_shape = new_qoi.posterior_sampler.sample_shape  # pyright: ignore[reportAttributeAccessIssue]
    sampler = IndexSampler(sample_shape)
    # We need to do the following work around to make IndexSampler compatible with TransformedPosterior. IndexSampler is
    # only designed to work with EnsembelPosterior, but when using outcome transforms on the model, the
    # EnsembelPosterior gets wrapped in a TransformedPosterior. This only causes issues when base samples are created
    # internally, so we create them here instead.
    # TODO(sw 2024-11-22): raise a ticket with botorch to fix this issue?
    sampler.register_buffer("base_samples", torch.zeros(sample_shape, dtype=torch.int))

    new_qoi.posterior_sampler = sampler

    return new_qoi(model=det_model)
