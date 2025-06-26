"""System test of qoi.GPBruteForce by running increasingly realistic version of the QoI and checking consistency.

`test_qoi_brute_force_system_test` is the function which orchestrates the test. The following function orchestrate the
individual steps, and contain additional detail regarding the motivation for the test.
- ground_truth_estimate
- qoi_no_gp
- qoi_gp_deterministic
- qoi_gp_low_uncertainty
- qoi_gp_high_uncertainty

This script is designed to be run interactively as well as though pytest.
"""

# ruff: noqa: T201
# pyright: reportUnnecessaryTypeIgnoreComment=false

# %%
import json
import time
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch
from ax import Models
from ax.modelbridge.torch import TorchModelBridge
from ax.models.torch.botorch_modular.model import BoTorchModel
from botorch.models import SingleTaskGP
from botorch.models.deterministic import GenericDeterministicModel, PosteriorMeanModel
from botorch.sampling.index_sampler import IndexSampler
from matplotlib.axes import Axes
from numpy.typing import NDArray
from torch.utils.data import DataLoader, Dataset, RandomSampler
from tqdm import tqdm

from axtreme.data import BatchInvariantSampler2d, MinimalDataset
from axtreme.eval import utils
from axtreme.eval.qoi_helpers import plot_col_histogram, plot_distribution, plot_groups
from axtreme.eval.qoi_job import QoIJob, QoIJobResult
from axtreme.experiment import add_sobol_points_to_experiment
from axtreme.qoi import GPBruteForce
from axtreme.sampling import NormalIndependentSampler
from axtreme.utils import population_estimators, transforms

# If running interactively, this is required to find examples
if __name__ == "__main__":
    import sys

    root_dir = Path(__file__).parent.parent.parent
    sys.path.append(str(root_dir))

from examples.demo2d.problem import brute_force, env_data, simulator
from examples.demo2d.problem.experiment import make_experiment

# %%
torch.set_default_dtype(torch.float64)

# %%
# TODO(sw 2024-11-19): To fit better with pytest this should be global
N_ENV_SAMPLES_PER_PERIOD = 1000

_: Any


# %% Prep plotting function to include brute force
def plot_best_guess(df: pd.DataFrame, ax: Axes, brute_force: float | None = None) -> None:
    """plot_col_histogram with SE added around the mean.

    This function conforms to the interface in axtreme.eval.qoi_plots
    """
    plot_col_histogram(df, ax, brute_force=brute_force)

    # Add the se plot
    ax_twin = ax.twinx()
    samples = torch.tensor(df.loc[:, "mean"].to_numpy())
    sample_mean_se_dist = population_estimators.sample_mean_se(samples)
    _ = population_estimators.plot_dist(sample_mean_se_dist, ax=ax_twin, c="red", label="dist mean 99.7% conf interval")
    _ = ax_twin.legend()


def best_guess_mean_vs_true_value(df: pd.DataFrame, true_value: float) -> torch.Tensor:
    """Helper to return the z score of the true value under the sample mean distribution.

    Args:
        df: Dataframe expected to have column 'mean'
        true_value: the population mean
    """
    samples = torch.tensor(df.loc[:, "mean"].to_numpy())
    sample_mean_se_dist = population_estimators.sample_mean_se(samples)

    return (true_value - sample_mean_se_dist.mean) / sample_mean_se_dist.stddev


def collect_statistics(df: pd.DataFrame, true_value: float) -> dict[str, float]:
    """Helper to return a dictionary of statistics for a dataframe.

    Args:
        df: expected to contain columns 'mean', 'var' and 'samples'.
        true_value: the true value of the population mean.

    Returns:
        Dictionary with the following format:

        {best_guess_z: z score of the true value under the sample mean distribution,
        best_guess_mean: mean of the sample mean distribution,
        best_guess_std: standard deviation of the sample mean distribution,
        var_mean: mean of the variance distribution,
        var_std: standard deviation of the variance distribution}
    """
    return {
        "best_guess_z": float(best_guess_mean_vs_true_value(df, true_value=true_value)),
        "best_guess_mean": df["mean"].mean(),
        "best_guess_std": df["mean"].std(),
        "var_mean": df["var"].mean(),
        "var_std": df["var"].std(),
    }


def get_id() -> str:
    """Produces and id which can be used to label related objects when saving."""
    import git

    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha

    return datetime.now().strftime("%y_%m_%d-%H_%M_%S") + f"__commit_{sha}"  # noqa: DTZ005


@pytest.fixture(scope="module")
def output_dir():
    output_dir = Path(__file__).parent / "results" / "gp_bruteforce" / get_id()
    output_dir.mkdir(parents=True)
    return output_dir


# To get more information run with `uv run pytest -m system -s``
@pytest.mark.system
@pytest.mark.non_deterministic
def test_qoi_brute_force_system_test(  # noqa: C901, PLR0913, PLR0912, PLR0915
    output_dir: None | Path,
    n_periods: int = 101,
    n_posterior_samples: int = 50,
    n_qoi_runs: int = 50,
    jobs_input_file: None | Path | pd.DataFrame = None,
    error_tol_scaling: float = 1,
    *,
    show_plots: bool = False,
    run_tests: bool = True,
    return_statistics: bool = False,
) -> dict[str, dict[str, float]] | None:
    """System test of QOI_brute_force by running increasingly realistic version of the QoI and checking consistency.

    Args:
        n_periods: number of periods to use. each produces `n_qoi_runs` estimates of the ERD.
        n_posterior_samples: number of posterior samples to take. Each produces an estimate of the QoI.
        n_qoi_runs: Number of times to run the QoIEstimator (a single run produces n_posterior_sampler estimates)
        jobs_input_file: path to json file contain QoIJobResult files.
            - if provided, will be used for testing (This is useful for analysis)
            - if None, jobs will be created and run, and then testing will occure.
        output_dir: directory to save plots to. If none, will not save output.
        error_tol_scaling: The error allowed in assertions is multiplied by this number.
        show_plots: If True plots will be shown.
        run_tests: If True will run assertions on the statistics produced by each qoi.
        return_statistics: If ture the function will return the statistic calculated.

    Details:
    Testing undertakes the following steps:
    - Ground truth estimate: use underlying function directly to make a QoI estimate.
        - Purpose: Gives an estimate of the confidence that can be achieved with a given number of samples. We then
        compare future results to this.
    - QoI with true underlying: use the QoI methodology with the true underling.
        - Purpose: confirm that the QoI implementation (excluding the GP) is correct (matches ground truth).
        - Purpose: Check setting for the number of periods used.
    - QoI with no variance GP: Using a GP with many points, remove uncertainty and calculate QoI.
        - Purpose: Check the quality of fit of the posterior mean. Introduce GP without posterior effects.
    - QoI with minimal variance GP: Using a GP with many points (and therefor very small uncertainty).
        - Purpose: Confirm produces the same results as above after posterior is introduced.
    - QoI with uncertain GP: Use a gp with only a few training points.
        - Purpose: Explore how posterior samples performs in these conditions.
        - Expectations: Higher variance withing a QoI output, potentially introduces bias

    Using this script to test a new QoIEstimator:
    - Start by producing a ground truth: This give you:
        - best guess (and uncertainty in this should expect)
        - variance in the QoI dist that should be expected.
    - Use the QoI using true underlying: This gives you:
        - Confirmation that with a perfect knowledge of the underling function the ground truth can be reporduced
    - Introduce the GP:
        - check using well trained GP it is possible to reproduce the ground truth

    NOTE: the bounds set on the test have been identified empirically. They should only fail 1% off the time. The method
    for obtaining these bounds is at the end of this file.

    Return:
    Nested dictionary.

        {"<QoI_name>": collect_statistics(df)}
    """
    #### Problem parameters
    """Note:
    It is useful to compare models with the same 'budget' of estimates. The budget is primarily a measure of how many
    samples are drawn from the surrogate. e.g in the ground truth the budget for a single QoI is:

    budget = n_periods * N_ENV_SAMPLES_PER_PERIOD * n_posterior_samples

    Use comparable budgets allows apples-to-apples comparison of the confidence and variance of different methods.
    """
    # bookkeeping parameters
    jobs_output_file = (
        output_dir / "qoi_job_results.json" if output_dir else None
    )  # Problem constants # TODO(sw): come back with a cleaner way to do this
    brute_force_qoi: float = float(
        brute_force.collect_or_calculate_results(period_length=N_ENV_SAMPLES_PER_PERIOD, num_estimates=300_000)[
            0
        ].median()
    )

    _data = env_data.collect_data()
    env_dataset: Dataset[NDArray[np.float64]] = MinimalDataset(_data.to_numpy())

    ### Helpers:
    # plotting helpers
    plot_distribution_bf = partial(plot_distribution, brute_force=brute_force_qoi)
    plot_dist_variance = partial(plot_col_histogram, col_name="var")
    plot_best_guess_bf = partial(plot_best_guess, brute_force=brute_force_qoi)

    # If passed existing jobs, these should be used instead.
    # This can be somewhat frail, requires consitent naming between rungs
    if isinstance(jobs_input_file, Path):
        with jobs_input_file.open() as fp:
            data = json.load(fp)
            df_jobs = pd.json_normalize(data, max_level=1)
            df_jobs.columns = df_jobs.columns.str.removeprefix("tags.")
    elif isinstance(jobs_input_file, pd.DataFrame):
        df_jobs = jobs_input_file

    # collect results
    statistics = {}

    ##### Run the differest QoIEstimator
    """Ground truth testing:
    Expections:
    - Should be a very good (unbiased estimate) for the true values
    """
    start_time = time.time()
    if jobs_input_file is not None:
        df_ground_truth = df_jobs[df_jobs["name"] == "ground_truth"]
    else:
        df_ground_truth = ground_truth_estimate(n_periods, n_posterior_samples, n_qoi_runs, jobs_output_file)

    statistics["ground_truth"] = collect_statistics(df_ground_truth, brute_force_qoi)

    # Plot results
    fig_ground_truth = plot_groups(
        df_ground_truth.groupby("name"), [plot_distribution_bf, plot_best_guess_bf, plot_dist_variance]
    )
    fig_ground_truth.savefig(str(output_dir / "ground_truth.png")) if output_dir else None
    fig_ground_truth.show() if show_plots else None

    ## Ground truth tests:
    if run_tests:
        stats_ground_truth = statistics["ground_truth"]
        assert abs(stats_ground_truth["best_guess_z"]) < 4.3 * error_tol_scaling
    print(f"Ground truth {(time.time() - start_time) // 60:.0f}:{time.time() - start_time:.2f}")

    ##### Qoi_no_gp
    """QoI_no_gp testing
    Expections:
    - Should be a very good (unbiased estimate) for the true values.
    - best guess variance should be very similar
    - variance distribution should be similar
    """
    start_time = time.time()
    if jobs_input_file is not None:
        df_no_gp = df_jobs[df_jobs["name"] == "qoi_no_gp"]
    else:
        df_no_gp = qoi_no_gp(env_dataset, n_periods, n_posterior_samples, n_qoi_runs, jobs_output_file)

    statistics["qoi_no_gp"] = collect_statistics(df_no_gp, brute_force_qoi)

    # plot results
    fig_qoi_no_gp = plot_groups(
        df_no_gp.groupby("name"), [plot_distribution_bf, plot_best_guess_bf, plot_dist_variance]
    )
    fig_qoi_no_gp.savefig(str(output_dir / "qoi_no_gp.png")) if output_dir else None
    fig_qoi_no_gp.show() if show_plots else None

    # Run tests
    if run_tests:
        # fmt: off
        stats_no_gp = statistics["qoi_no_gp"]
        assert abs(stats_no_gp["best_guess_z"]) < 4 * error_tol_scaling
        assert stats_no_gp["best_guess_std"] == pytest.approx(stats_ground_truth["best_guess_std"], rel=0.35 * error_tol_scaling)  # noqa: E501
        assert stats_no_gp["var_mean"] == pytest.approx(stats_ground_truth["var_mean"], rel=0.15 * error_tol_scaling)
        assert stats_no_gp["var_std"] == pytest.approx(stats_ground_truth["var_std"], rel=0.5 * error_tol_scaling)
        # fmt: on

    print(f"QoI_no_gp {(time.time() - start_time) // 60:.0f}:{time.time() - start_time:.2f}")
    ##### Highly trained GP
    """Deterministic GP
    Expections:
        - Results should be very similar to brute force, potentially we some added bias due to gp fit.
            - Only interested in understanding this bias
    """
    start_time = time.time()
    if jobs_input_file is not None:
        df_gp_deterministic = df_jobs[df_jobs["name"] == "qoi_gp_deterministic"]
    else:
        df_gp_deterministic = qoi_gp_deterministic(
            env_dataset, n_periods, n_posterior_samples, n_qoi_runs, jobs_output_file
        )

    statistics["qoi_gp_deterministic"] = collect_statistics(df_gp_deterministic, brute_force_qoi)

    # plot results
    fig_qoi_gp_deterministic = plot_groups(
        df_gp_deterministic.groupby("name"), [plot_distribution_bf, plot_best_guess_bf, plot_dist_variance]
    )
    fig_qoi_gp_deterministic.savefig(str(output_dir / "qoi_gp_deterministic.png")) if output_dir else None
    fig_qoi_gp_deterministic.show() if show_plots else None

    # Run tests
    if run_tests:
        stats_gp_deterministic = statistics["qoi_gp_deterministic"]
        assert abs(stats_gp_deterministic["best_guess_z"]) < 5 * error_tol_scaling

    print(f"Deterministic GP {(time.time() - start_time) // 60:.0f}:{time.time() - start_time:.2f}")

    """Gp low variance

    The GP has a very low amount of uncertianty. It is expected to produce a very similar result as above.
    Expected results:
    - produces results very close to the ground truth.
    - Slightly larger variance in best guess (provided gp_deterministic is a good/unbiased estimator)
    """
    start_time = time.time()
    if jobs_input_file is not None:
        df_gp_low_uncertainty = df_jobs[df_jobs["name"] == "qoi_gp_low_uncertainty"]
    else:
        df_gp_low_uncertainty = qoi_gp_low_uncertainty(
            env_dataset, n_periods, n_posterior_samples, n_qoi_runs, jobs_output_file
        )

    statistics["qoi_gp_low_uncertainty"] = collect_statistics(df_gp_low_uncertainty, brute_force_qoi)

    # plot results
    fig_qoi_gp_low_uncertainty = plot_groups(
        df_gp_low_uncertainty.groupby("name"), [plot_distribution_bf, plot_best_guess_bf, plot_dist_variance]
    )
    fig_qoi_gp_low_uncertainty.savefig(str(output_dir / "qoi_gp_low_uncertainty.png")) if output_dir else None
    fig_qoi_gp_low_uncertainty.show() if show_plots else None

    # run tests
    if run_tests:
        stats_low_uncertainty = statistics["qoi_gp_low_uncertainty"]
        # fmt: off
        assert abs(stats_low_uncertainty["best_guess_z"]) < 5 * error_tol_scaling
        assert stats_low_uncertainty["best_guess_std"] == pytest.approx(stats_ground_truth["best_guess_std"], rel=0.6 * error_tol_scaling)  # noqa: E501
        assert stats_low_uncertainty["var_mean"] == pytest.approx(stats_ground_truth["var_mean"], rel=0.15 * error_tol_scaling)  # noqa: E501
        assert stats_low_uncertainty["var_std"] == pytest.approx(stats_ground_truth["var_std"], rel=0.5 * error_tol_scaling)  # noqa: E501
        # fmt: on

    print(f"Gp low variance {(time.time() - start_time) // 60:.0f}:{time.time() - start_time:.2f}")
    """Gp high variance

    The GP has a larger amount of uncertianty. It is expected to produce moer uncertain results.

    Expected results:
    - There are no strict tests that can be performed here. See method for details.
    """
    start_time = time.time()
    if jobs_input_file is not None:
        df_qoi_gp_high_uncertainty = df_jobs[df_jobs["name"] == "qoi_gp_high_uncertainty"]
    else:
        df_qoi_gp_high_uncertainty = qoi_gp_high_uncertainty(
            env_dataset, n_periods, n_posterior_samples, n_qoi_runs, jobs_output_file
        )

    statistics["qoi_gp_high_uncertainty"] = collect_statistics(df_qoi_gp_high_uncertainty, brute_force_qoi)

    fig_qoi_gp_high_uncertainty = plot_groups(
        df_qoi_gp_high_uncertainty.groupby("name"), [plot_distribution_bf, plot_best_guess_bf, plot_dist_variance]
    )
    fig_qoi_gp_high_uncertainty.savefig(str(output_dir / "qoi_gp_high_uncertainty.png")) if output_dir else None
    fig_qoi_gp_high_uncertainty.show() if show_plots else None
    print(f"Gp high variance {(time.time() - start_time) // 60:.0f}:{time.time() - start_time:.2f}")

    # TODO(sw 2024-12-9): This is a hacky fix so statistics are easily available when calibrating bounds (see bottom of
    # file), and they are not returned in general (when this is running through pytest). Statistic should probably be
    # saved with the plots as well.
    if return_statistics:
        return statistics

    return None


# %%
def qoi_gp_high_uncertainty(
    env_dataset: Dataset[NDArray[np.float64]],
    n_periods: int,
    n_posterior_samples: int,
    n_qoi_runs: int,
    jobs_output_file: None | Path = None,
) -> pd.DataFrame:
    """Run the QoI with a highly trained model.

    Args:
        env_dataset: Dataset containing the envdata we have access to.
        n_periods: number of periods to use. each produces `n_qoi_runs` estimates of the ERD.
        n_posterior_samples: number of posterior samples to take. Each produces an estimate of the QoI.
        n_qoi_runs: Number of times to run the QoIEstimator (a single run produces n_posterior_sampler estimates)
        jobs_output_file (None | Path): Json file to write raw run results to. If not provided do no write results.

    Return:
        A DataFrame where each row represents a single QoIeEstimator run. Contains columns:
            - mean: The samples mean produced by the QoIeEstimator for that run.
            - var: The samples var produced by the QoIeEstimator for that run.
            - samples: The samples produced by the QoIeEstimator for that run.
            - name: Name of the estimator. This is used to group these results together.

    Details:
        As described in `qoi_gp_low_uncertainty`:
        - Can't check this value against the brute force because imperfect GP will have bias.
        - Can't compared to the PosteriorMean of the GP, as there can be non-linear response.

    Expectations:
        General:
        - Much greater variance within a QoI estimate (each n_est uses a very different underling function)

        Increase the posterior samples will:
        - Reduce the variance in best guesses
        - Wont' reduce the mean of the variance, but should reduce the variance of the variance
    """
    botorch_model = get_trained_gp(n_points=20)

    qoi_gp_high_uncertain_jobs = []
    for i in range(n_qoi_runs):
        qoi_est = GPBruteForce(
            # random dataloader give different env samples for each instance
            env_iterable=get_dataloader(env_dataset, n_periods, batch_size=64),
            # Created new each time to insure not freezing/subsample.
            # because we are not using a deterministic sampler we need this (real) sampler
            posterior_sampler=NormalIndependentSampler(torch.Size([n_posterior_samples])),
            no_grad=True,
        )

        qoi_gp_high_uncertain_jobs.append(
            QoIJob(
                name=f"qoi_gp_high_uncertainty_{i}",
                qoi=qoi_est,
                model=botorch_model,
                tags={"name": "qoi_gp_high_uncertainty"},
            )
        )

    qoi_gp_high_uncertain_results = [job(output_file=jobs_output_file) for job in tqdm(qoi_gp_high_uncertain_jobs)]
    df_jobs = pd.json_normalize([item.to_dict() for item in qoi_gp_high_uncertain_results], max_level=1)
    df_jobs.columns = df_jobs.columns.str.removeprefix("tags.")

    return df_jobs


def qoi_gp_low_uncertainty(
    env_dataset: Dataset[NDArray[np.float64]],
    n_periods: int,
    n_posterior_samples: int,
    n_qoi_runs: int,
    jobs_output_file: None | Path = None,
) -> pd.DataFrame:
    """Run the QoI with a highly trained model.

    Args:
        env_dataset: Dataset containing the envdata we have access to.
        n_periods (int): number of periods to use. each produces `n_qoi_runs` estimates of the ERD.
        n_posterior_samples (int): number of posterior samples to take. Each produces an estimate of the QoI.
        n_qoi_runs (int): Number of times to run the QoIEstimator (a single run produces n_posterior_sampler estimates)
        jobs_output_file (None | Path): Json file to write raw run results to. If not provided do no write results.

    Return:
        A DataFrame where each row represents a single QoIeEstimator run. Contains columns:
            - mean: The samples mean produced by the QoIeEstimator for that run.
            - var: The samples var produced by the QoIeEstimator for that run.
            - samples: The samples produced by the QoIeEstimator for that run.
            - name: Name of the estimator. This is used to group these results together.

    Details:
        The GP has a very low amount of uncertianty. Expected to produce a very similar result as Gp deterministic.

        Note: Uncertinaty in the GP can (correctly) produce bias results. This can occur if the response calcuated from
        the GP is very non-linear/sensitive to the GP. e.g:
        - Sample smaller than the posterior median: result = .1
        - Sample at posterior mean: result = 1
        - Sample larger than the posterior median: result = 10
        When taking the best guess from these results (the mean = 3.7):
        - this is a poor representation of the posterior mean
        - this is a good representation of the response you could get due to uncertainty in the underlying function.

        Expected results:
        - produces results very close to the ground truth.
        - Slightly larger variance is best guess.
    """
    botorch_model = get_trained_gp()

    dataloader = get_dataloader(env_dataset, n_periods, batch_size=64)
    qoi_gp_uncertain_jobs = []
    for i in range(n_qoi_runs):
        qoi_est = GPBruteForce(
            # random dataloader give different env samples for each instance
            env_iterable=dataloader,
            # Created new each time to insure not freezing/subsample.
            # because we are not using a deterministic sampler we need this (real) sampler
            posterior_sampler=NormalIndependentSampler(torch.Size([n_posterior_samples])),
            no_grad=True,
        )

        qoi_gp_uncertain_jobs.append(
            QoIJob(
                name=f"qoi_gp_low_uncertainty_{i}",
                qoi=qoi_est,
                model=botorch_model,
                tags={"name": "qoi_gp_low_uncertainty"},
            )
        )

    qoi_gp_uncertain_results = [job(output_file=jobs_output_file) for job in tqdm(qoi_gp_uncertain_jobs)]
    df_jobs = pd.json_normalize([item.to_dict() for item in qoi_gp_uncertain_results], max_level=1)
    df_jobs.columns = df_jobs.columns.str.removeprefix("tags.")

    return df_jobs


def qoi_gp_deterministic(
    env_dataset: Dataset[NDArray[np.float64]],
    n_periods: int,
    n_posterior_samples: int,
    n_qoi_runs: int,
    jobs_output_file: None | Path = None,
):
    """Run the QoI with a highly trained, deterministic model.

    Args:
        env_dataset: Dataset containing the envdata we have access to.
        n_periods (int): number of periods to use. each produces `n_qoi_runs` estimates of the ERD.
        n_posterior_samples (int): number of posterior samples to take. Each produces an estimate of the QoI.
        n_qoi_runs (int): Number of times to run the QoIEstimator (a single run produces n_posterior_sampler estimates)
        jobs_output_file (None | Path): Json file to write raw run results to. If not provided do no write results.

    Return:
        A DataFrame where each row represents a single QoIeEstimator run. Contains columns:
            - mean: The samples mean produced by the QoIeEstimator for that run.
            - var: The samples var produced by the QoIeEstimator for that run.
            - samples: The samples produced by the QoIeEstimator for that run.
            - name: Name of the estimator. This is used to group these results together.

    Details:
        We want to check the GP is a good fit, because if not the results will be bias regardless of other settings.
        - we start by checking if the posterior mean is a good fit.

    Expected results:
        We would expect this to produce a very similar result to the `qoi_no_gp`.
        - produces the same results as the ground truth.
        - May have slight bias due to slight model misfit.
    """
    botorch_model = get_trained_gp()

    gp_deterministic = PosteriorMeanModel(botorch_model)

    dataloader = get_dataloader(env_dataset, n_periods, batch_size=64)
    qoi_gp_deterministic_jobs = []
    for i in range(n_qoi_runs):
        qoi_est = GPBruteForce(
            # random dataloader give different env samples for each instance
            env_iterable=dataloader,
            # IndexSampler needs to be used with GenericDeterministicModel. Each sample just selects the mean.
            posterior_sampler=IndexSampler(torch.Size([n_posterior_samples])),
            no_grad=True,
        )

        qoi_gp_deterministic_jobs.append(
            QoIJob(
                name=f"qoi_gp_deterministic_{i}",
                qoi=qoi_est,
                model=gp_deterministic,
                tags={"name": "qoi_gp_deterministic"},
            )
        )

    qoi_gp_deterministic_results = [job(output_file=jobs_output_file) for job in tqdm(qoi_gp_deterministic_jobs)]
    df_jobs = pd.json_normalize([item.to_dict() for item in qoi_gp_deterministic_results], max_level=1)
    df_jobs.columns = df_jobs.columns.str.removeprefix("tags.")

    return df_jobs


def get_trained_gp(n_points: int = 512) -> SingleTaskGP:
    """Create a botorch model."""
    # TODO(sw): would be good to pull this out of here
    exp = make_experiment()
    add_sobol_points_to_experiment(exp, n_iter=n_points, seed=7)

    botorch_model_bridge = Models.BOTORCH_MODULAR(
        experiment=exp,
        data=exp.fetch_data(),
    )
    assert isinstance(botorch_model_bridge, TorchModelBridge)
    input_transform, outcome_transform = transforms.ax_to_botorch_transform_input_output(
        transforms=list(botorch_model_bridge.transforms.values()), outcome_names=botorch_model_bridge.outcomes
    )
    ax_model = botorch_model_bridge.model
    assert isinstance(ax_model, BoTorchModel)
    botorch_model = ax_model.surrogate.model
    assert isinstance(botorch_model, SingleTaskGP)
    botorch_model.outcome_transform = outcome_transform
    botorch_model.input_transform = input_transform

    return botorch_model


def qoi_no_gp(
    env_dataset: Dataset[NDArray[np.float64]],
    n_periods: int,
    n_posterior_samples: int,
    n_qoi_runs: int,
    jobs_output_file: None | Path = None,
) -> pd.DataFrame:
    """Test the QoIEstimator with the true underlying function.

    Args:
        env_dataset: Dataset containing the envdata we have access to.
        n_periods (int): number of periods to use. each produces `n_qoi_runs` estimates of the ERD.
        n_posterior_samples (int): number of posterior samples to take. Each produces an estimate of the QoI.
        n_qoi_runs (int): Number of times to run the QoIEstimator (a single run produces n_posterior_sampler estimates)
        jobs_output_file (None | Path): Json file to write raw run results to. If not provided do no write results.

    Return:
        A DataFrame where each row represents a single QoIeEstimator run. Contains columns:
            - mean: The samples mean produced by the QoIeEstimator for that run.
            - var: The samples var produced by the QoIeEstimator for that run.
            - samples: The samples produced by the QoIeEstimator for that run.
            - name: Name of the estimator. This is used to group these results together.

    Details:
    When using the true underlying (rather than a GP) there are still difference between the QoIEstimator and the
    Ground truth. This is caused by the difference in how env data is used:
    - Ground truth: each period gets a unique sample of env data
        - (e.g there are n_erd_samples * n_ests unique periods of env data)
    - QoIEstimator: All estimates produce by a single run see the same set of periods
        - (e.g there are n_erd_samples unique periods of env data)

    If insuffeciently large n_periods is used:
    - best guess will have more noise, (individual samples will be more biased), the overall there should remain no bias
    - the bounds on the distribution will be smaller than the real value

    This checks that this value is approapriately set. If not, the other QoIEstimator (e.g using Gps etc) will not
    perform.

    Expected result:
    - Produces the same results as the ground truth.
    """

    def true_scale_func_tensor(x: torch.Tensor) -> torch.Tensor:
        return torch.from_numpy(simulator._true_scale_func(x.numpy()))

    def true_loc_func_tensor(x: torch.Tensor) -> torch.Tensor:
        return torch.from_numpy(simulator._true_loc_func(x.numpy()))

    def true_underling_func(x: torch.Tensor) -> torch.Tensor:
        """Put true underling function into format required for GenericDeterministicModel.

        Args:
        x: tensor of dimensions (*b, n, d)

        returns:
        tensor of dimensions (*b, n, d)
        """
        locs = true_loc_func_tensor(x).unsqueeze(-1)
        scales = true_scale_func_tensor(x).unsqueeze(-1)
        return torch.concat([locs, scales], dim=-1)

    # This is roughly equivalent to useing a GP the no variance
    # Running this model produces  botorch.posteriors.ensemble.EnsemblePosterior. They require index samplers
    det_model = GenericDeterministicModel(true_underling_func, num_outputs=2)
    dataloader = get_dataloader(env_dataset, n_periods)
    qoi_no_gp_jobs = []
    for i in range(n_qoi_runs):
        qoi_est = GPBruteForce(
            # random dataloader give different env samples for each instance
            env_iterable=dataloader,
            # IndexSampler needs to be used with GenericDeterministicModel. Each sample just selects the mean.
            posterior_sampler=IndexSampler(torch.Size([n_posterior_samples])),
            no_grad=True,
        )

        qoi_no_gp_jobs.append(QoIJob(name=f"qoi_no_gp_{i}", qoi=qoi_est, model=det_model, tags={"name": "qoi_no_gp"}))

    qoi_no_gp_results = [job(output_file=jobs_output_file) for job in tqdm(qoi_no_gp_jobs)]

    df_jobs = pd.json_normalize([item.to_dict() for item in qoi_no_gp_results], max_level=1)
    df_jobs.columns = df_jobs.columns.str.removeprefix("tags.")
    return df_jobs


def get_dataloader(dataset: Dataset[NDArray[np.float64]], n_periods: int, batch_size: int = 64):
    """Helper to create a Dataloader which randomly samples from the dataset.

    Dataloader samples with replacement from the dataset. Total iterable will have shape
    (n_periods, N_ENV_SAMPLES_PER_PERIOD). If the dataloader is reused, it will generate new random samples.

    Args:
        dataset: The available data to draw samples from
        n_periods: Number of periods of data to produce.
        batch_size: N_ENV_SAMPLES_PER_PERIOD will be split into batches of this size
    """
    generator = torch.Generator()
    _ = generator.manual_seed(7)
    replacement_sampler = RandomSampler(
        data_source=dataset,  # type: ignore[arg-type]
        num_samples=n_periods * N_ENV_SAMPLES_PER_PERIOD,
        generator=generator,
        replacement=True,
    )
    batch_sampler_rand = BatchInvariantSampler2d(
        sampler=replacement_sampler, batch_shape=torch.Size([n_periods, batch_size])
    )
    return DataLoader(dataset, batch_sampler=batch_sampler_rand)


def ground_truth_estimate(
    n_periods: int, n_posterior_samples: int, n_qoi_runs: int, jobs_output_file: None | Path = None
) -> pd.DataFrame:
    """For a given budget of ERD samples, what answer is achieved using the true undelying directly (perfect knowledge).

    If other QoI outputs have:
    - More noise: likely need to increase the samples of different source of randomness (e.g gp)
    - Less noise: likely have frozen/subsampled sources of randomness

    The behaviour of the mean and variance of each distribution can help analyse these issues.

    Args:
        n_periods (int): number of periods to use. each produces `n_qoi_runs` estimates of the ERD.
        n_posterior_samples (int): number of posterior samples to take. Each produces an estimate of the QoI.
        n_qoi_runs (int): Number of times to run the QoIEstimator (a single run produces n_posterior_sampler estimates)
        jobs_output_file (None | Path): Json file to write raw run results to. If not provided do no write results.

    For each sample from the GP (e.g a potential underling):
        - Number of ERD samples = n_periods * n_qoi_runs

    Return:
        A DataFrame where each row represents a single QoIeEstimator run. Contains columns:
            - mean: The samples mean produced by the QoIeEstimator for that run.
            - var: The samples var produced by the QoIeEstimator for that run.
            - samples: The samples produced by the QoIeEstimator for that run.
            - name: Name of the estimator. This is used to group these results together.
    """
    brute_force_erd_samples, _ = brute_force.collect_or_calculate_results(
        period_length=N_ENV_SAMPLES_PER_PERIOD, num_estimates=300_000
    )

    # Turn this into job results to we can plot in the same manner as above
    ground_truth_results = []
    for _ in range(n_qoi_runs):
        qoi_comparable_output = qoi_comparable_output_using_full_brute_force(
            n_erd_samples=n_periods, n_ests=n_posterior_samples, brute_force_erd_samples=brute_force_erd_samples
        )
        # Directly skip to results here as nothing is required to run.
        ground_truth_results.append(
            QoIJobResult(
                mean=qoi_comparable_output.mean(),
                var=qoi_comparable_output.var(),
                samples=qoi_comparable_output,
                tags={"name": "ground_truth"},
            )
        )

    if jobs_output_file:
        for result in ground_truth_results:
            utils.append_to_json(result.to_dict(), jobs_output_file)

    df_jobs = pd.json_normalize([item.to_dict() for item in ground_truth_results], max_level=1)
    df_jobs.columns = df_jobs.columns.str.removeprefix("tags.")
    return df_jobs


def qoi_comparable_output_using_full_brute_force(
    n_erd_samples: int, n_ests: int, brute_force_erd_samples: torch.Tensor
) -> torch.Tensor:
    """Mimics the output the QoIEstimator produces, but uses precalculated ERD samples directly.

    Purpose is to show what output would expect from a 'perfect' QoIEstimator with this number of samples.
        - 'perfect' Meaning no GP uncertainty, and no subsampling/freezing. Achieved because precalc results:
            - use the true underling function
            - use a unique sample of weather for each ERD sample/

    Args:
        n_erd_samples: the number of erd samples used to make a single estimate of the QoI.
        - Note: in QoIestimate the number of ERD samples is controlled by n_periods.
        n_ests: The number of QoI estimates to make.
        - Note: in QoIestimate the number of estimates made is controlle by n_posterior_samples
        brute_force_erd_samples: (n,) precalculated ERD samples. Usecases often contain methods for bruteforce.
            - Should be at least: n > n_erd_samples * n_ests
            - ideally n > n_erd_samples * n_ests * n_qoi_runs

    Return:
        `n_ests` for the QoI. This output is comparable with the output returned by the QoIEstimator interface.
    """
    sample_idx = torch.randint(0, len(brute_force_erd_samples), size=torch.Size([n_ests, n_erd_samples]))
    samples = brute_force_erd_samples[sample_idx]
    return samples.median(dim=-1).values  # noqa: PD011


# %%
if __name__ == "__main__":
    # %%
    test_qoi_brute_force_system_test(output_dir=None, n_qoi_runs=120, run_tests=True, show_plots=True)  # type: ignore  # noqa: PGH003

    # %%
    import matplotlib.pyplot as plt

    _ = plt.ioff()  # stop plots from displaying for repeate runs
    """This following is a helper to determine the bounds testing in qoi_estimator_output.

    Here we first run the test with a large number of samples, then with subsampling estimate what the bounds would be
    if running with fewer samples. Now that the bounds are calibrated, we can run the tests with fewer samples,
    which takes less time.

    The following is the process for calibrating the bounds in the estimators or number of subsamples changes
    """
    # %% Args to set
    total_number_of_runs = 400
    subsample_size = 50

    # %%
    # Make a large number of samples once off:
    output_dir_path = Path("results") / "gp_bruteforce" / get_id()
    output_dir_path.mkdir(exist_ok=True)

    _ = test_qoi_brute_force_system_test(
        output_dir=output_dir_path,
        n_qoi_runs=total_number_of_runs,  # make as large as feasible.
        run_tests=False,
    )

    # %%
    # read the Qoi_jons
    with (output_dir_path / "qoi_job_results.json").open("r") as fp:
        qoi_job_results = json.load(fp)

    full_df = pd.json_normalize(qoi_job_results, max_level=1)
    full_df.columns = full_df.columns.str.removeprefix("tags.")

    # %% Subsample the results
    def sample_group(group, n=subsample_size):  # type: ignore  # noqa: ANN001, PGH003
        return group.sample(n=min(len(group), n))

    # Group by 'name' and sample 50 items from each group
    all_statistics = []
    for _ in range(100):
        sampled_df = full_df.groupby("name", group_keys=False).apply(sample_group)
        statistics = test_qoi_brute_force_system_test(
            output_dir=None, jobs_input_file=sampled_df, run_tests=False, show_plots=False, return_statistics=True
        )
        all_statistics.append(statistics)

    df = pd.json_normalize(all_statistics, max_level=1)  # type: ignore  # noqa: PD901, PGH003
    df.head()  # type: ignore  # noqa: PGH003

    # %% Get the statistic for each group.
    # fmt: off
    # do the statistics for the different tests
    # Ground truth
    print("best_guess_z: ",df["ground_truth.best_guess_z"].abs().max())

    # %%
    #  qoi_no_gp
    print("best_guess_z: ",df["qoi_no_gp.best_guess_z"].abs().max())

    print("best_guess_std\n",(df["qoi_no_gp.best_guess_std"] / df["ground_truth.best_guess_std"]).agg(["min","max"]))
    print("var_mean\n",(df["qoi_no_gp.var_mean"] / df["ground_truth.var_mean"]).agg(["min","max"]))
    print("var_std\n",(df["qoi_no_gp.var_std"] / df["ground_truth.var_std"]).agg(["min","max"]))


    # %% qoi_gp_deterministic
    print("best_guess_z: ", df["qoi_gp_deterministic.best_guess_z"].abs().max())


    # %% qoi_gp_low_uncertainty
    print("best_guess_z: ", df["qoi_gp_low_uncertainty.best_guess_z"].abs().max())

    print("best_guess_std\n",(df["qoi_gp_low_uncertainty.best_guess_std"] / df["ground_truth.best_guess_std"]).agg(["min","max"]))  # noqa: E501
    print("var_mean\n",(df["qoi_gp_low_uncertainty.var_mean"] / df["ground_truth.var_mean"]).agg(["min","max"]))
    print("var_std\n",(df["qoi_gp_low_uncertainty.var_std"] / df["ground_truth.var_std"]).agg(["min","max"]))

    # fmt:on
    # %%
