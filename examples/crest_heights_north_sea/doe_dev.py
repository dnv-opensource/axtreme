"""The script explores the sensitivity of DOE results to the sobol seed and for DOE until a given stopping criteria.

The sobol seed controls the initial dataset generated (used to train the GP),
and the baseline performance the DOE/acquisition function is compared to.
This runs using the QoI and Experiment define in doe.py and is intend as
an additional analysis of how the sobol seed effects the DOE process defined there.

Additionally, it runs the DOE until a given stopping criteria is met.
This is useful to see how the DOE process converges to the QoI estimate with a given error and uncertaionty and how the
uncertainty in the estimate decreases over time. ALso direct comparison between Sobol and Look-ahead DOE is made to see
the benefits of the Look-ahead DOE approach.
"""

# %%
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from ax import Experiment
from ax.core import GeneratorRun
from ax.modelbridge.registry import Models
from doe import (  # type: ignore[import-not-found]
    QOI_METRIC,
    brute_force_qoi,
    create_sobol_generator,
    look_ahead_generator_run,
    make_exp,
    run_trials,
)
from problem import N_SIMULATIONS_PER_POINT  # type: ignore[import-not-found]

from axtreme.plotting.doe import plot_qoi_estimates_from_experiment

# %% [markdown]
# As of 2025-05-14, this multiple DOE experiments with different seeds for the sobol generator to see the effects this
# has on the DOE. For now it is quick and dirty, but works for the purpose of testing how the sobol generator seeding
# affects the DOE results.


# %%
def run_multiple_seeded_experiments(
    n_experiments: int = 10,
    n_iter: int = 15,
    warm_up_runs: int = 3,
    save_dir: str = "./results/doe/",
    base_seed: int = 5,
    seed_step: int = 10,
) -> tuple[list[Experiment], list[Experiment]]:
    """Run multiple Sobol and Look-ahead experiments with different seeds.

    Usefull for testing the sensitivity of the DOE results to different sobol seed.

    Args:
        n_experiments: Number of experiments to run with different seeds
        n_iter: Number of DOE iterations for each experiment
        warm_up_runs: Number of warm-up runs for each experiment
        save_dir: Directory to save plots
        base_seed: Starting seed value
        seed_step: Step between seed values for more diversity

    Returns:
        Tuple containing lists of Sobol and Look-ahead experiments
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Generate seeds
    seeds = [base_seed + i * seed_step for i in range(n_experiments)]

    # Lists to store experiments
    sobol_experiments = []
    look_ahead_experiments = []

    print(f"Running {n_experiments} experiments with seeds: {seeds}")

    # Create and run experiments
    for i, seed in enumerate(seeds):
        print(f"\nExperiment {i + 1}/{n_experiments} with seed {seed}")

        # Create and run Sobol experiment
        print("Running Sobol experiment...")
        exp_sobol = make_exp()
        _ = exp_sobol.add_tracking_metric(QOI_METRIC)
        sobol = Models.SOBOL(search_space=exp_sobol.search_space, seed=seed)
        sobol_generator_run = create_sobol_generator(sobol)

        _ = run_trials(
            experiment=exp_sobol,
            warm_up_generator=sobol_generator_run,
            doe_generator=sobol_generator_run,
            warm_up_runs=warm_up_runs,
            doe_runs=n_iter,
        )
        sobol_experiments.append(exp_sobol)

        # Create and run Look-ahead experiment
        print("Running Look-ahead experiment...")
        exp_look_ahead = make_exp()
        _ = exp_look_ahead.add_tracking_metric(QOI_METRIC)
        sobol = Models.SOBOL(search_space=exp_look_ahead.search_space, seed=seed)

        _ = run_trials(
            experiment=exp_look_ahead,
            warm_up_generator=create_sobol_generator(sobol),
            doe_generator=look_ahead_generator_run,
            warm_up_runs=warm_up_runs,
            doe_runs=n_iter,
        )
        look_ahead_experiments.append(exp_look_ahead)

    # Create individual comparison plots
    create_comparison_plots(
        sobol_experiments=sobol_experiments,
        look_ahead_experiments=look_ahead_experiments,
        seeds=seeds,
        brute_force_qoi_estimate=brute_force_qoi,
        save_dir=save_dir,
    )

    # Create summary plot
    print("Creating summary plot...")
    create_summary_plot(
        sobol_experiments,
        look_ahead_experiments,
        n_iter,
        brute_force_qoi,
        save_path=f"{save_dir}/sobol_vs_lookahead_summary.png",
    )

    return sobol_experiments, look_ahead_experiments


def create_comparison_plots(
    sobol_experiments: list[Experiment],
    look_ahead_experiments: list[Experiment],
    seeds: list[int],
    brute_force_qoi_estimate: float,
    save_dir: str,
) -> None:
    """Create comparison plots for multiple experiments generated using run_multiple_seeded_experiments().

    Args:
        sobol_experiments: List of Sobol experiments
        look_ahead_experiments: List of Look-ahead experiments
        seeds: List of seeds used for experiments
        brute_force_qoi_estimate: Reference value from brute force estimation
        save_dir: Directory to save plots
    """
    print("Creating individual comparison plots...")
    n_experiments = len(seeds)
    fig, axes = plt.subplots(min(4, (n_experiments + 2) // 3), min(3, n_experiments), figsize=(20, 15))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for i, seed in enumerate(seeds):
        if i < len(axes):
            ax = axes[i]

            # Plot Sobol experiment
            ax = plot_qoi_estimates_from_experiment(sobol_experiments[i], name=f"Sobol (seed={seed})", ax=ax)

            # Plot Look-ahead experiment
            ax = plot_qoi_estimates_from_experiment(
                look_ahead_experiments[i], ax=ax, color="green", name=f"Look-ahead (seed={seed})"
            )

            # Add reference line
            _ = ax.axhline(
                brute_force_qoi_estimate, c="black", label="Brute force estimate" if i == 0 else "", linestyle="--"
            )

            # Add labels
            _ = ax.set_xlabel("Number of DOE iterations")
            _ = ax.set_ylabel("QoI Estimate")
            _ = ax.set_title(f"Seed {seed}: Sobol vs Look-ahead")

            # Add legend only to the first plot
            if i == 0:
                _ = ax.legend(loc="upper right")

    # Remove empty subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(f"{save_dir}/sobol_vs_lookahead_comparison.png", dpi=300)
    plt.close()


def create_summary_plot(
    sobol_experiments: list[Experiment],
    look_ahead_experiments: list[Experiment],
    n_iter: int,
    brute_force_qoi_estimate: float,
    save_path: str,
) -> None:
    """Create a summary plot comparing the average performance of Sobol and Look-ahead experiments.

    Args:
        sobol_experiments: List of Sobol experiments
        look_ahead_experiments: List of Look-ahead experiments
        n_iter: Number of DOE iterations
        brute_force_qoi_estimate: Reference value from brute force estimation
        save_path: Path to save the plot
    """
    _ = plt.figure(figsize=(12, 8))

    # Extract QoI values
    sobol_qoi_values = []
    lookahead_qoi_values = []

    for exp_sobol, exp_look_ahead in zip(sobol_experiments, look_ahead_experiments, strict=False):
        # Extract data
        sobol_data = exp_sobol.fetch_data().df
        lookahead_data = exp_look_ahead.fetch_data().df

        # Get QoI metric values
        sobol_qoi = sobol_data[sobol_data["metric_name"] == "QoIMetric"]["mean"].values
        lookahead_qoi = lookahead_data[lookahead_data["metric_name"] == "QoIMetric"]["mean"].values

        sobol_qoi_values.append(sobol_qoi)
        lookahead_qoi_values.append(lookahead_qoi)

    # Print diagnostic information
    print(f"Number of experiments: {len(sobol_qoi_values)}")
    for i, (s_qoi, l_qoi) in enumerate(zip(sobol_qoi_values, lookahead_qoi_values, strict=False)):
        print(f"Experiment {i}: Sobol length={len(s_qoi)}, Lookahead length={len(l_qoi)}")
    print(f"Expected length: {n_iter + 1}")

    # Find the most common length if experiments have inconsistent lengths
    sobol_lengths = [len(x) for x in sobol_qoi_values]
    lookahead_lengths = [len(x) for x in lookahead_qoi_values]
    most_common_length = max(set(sobol_lengths + lookahead_lengths), key=(sobol_lengths + lookahead_lengths).count)

    # Adjust arrays to use most common length
    sobol_array = np.array([x for x in sobol_qoi_values if len(x) == most_common_length])
    lookahead_array = np.array([x for x in lookahead_qoi_values if len(x) == most_common_length])

    # Ensure we have data to plot
    if len(sobol_array) == 0 or len(lookahead_array) == 0:
        print("WARNING: No valid data for summary plot with consistent lengths")
    else:
        # Calculate mean and std dev
        sobol_mean = np.mean(sobol_array, axis=0)
        sobol_std = np.std(sobol_array, axis=0)
        lookahead_mean = np.mean(lookahead_array, axis=0)
        lookahead_std = np.std(lookahead_array, axis=0)

        # Plot
        x = np.arange(len(sobol_mean))
        _ = plt.plot(x, sobol_mean, "b-", label="Sobol (average)")
        _ = plt.fill_between(x, sobol_mean - sobol_std, sobol_mean + sobol_std, alpha=0.2, color="blue")
        _ = plt.plot(x, lookahead_mean, "g-", label="Look-ahead (average)")
        _ = plt.fill_between(
            x, lookahead_mean - lookahead_std, lookahead_mean + lookahead_std, alpha=0.2, color="green"
        )

        # Add reference line
        _ = plt.axhline(brute_force_qoi_estimate, c="black", label="Brute force estimate", linestyle="--")

        _ = plt.xlabel("Number of DOE iterations")
        _ = plt.ylabel("QoI Estimate")
        _ = plt.title("Average Performance: Sobol vs Look-ahead (with std. deviation)")
        _ = plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# %%
if __name__ == "__main__":
    # Run experiments from command line or notebook
    sobol_exps, lookahead_exps = run_multiple_seeded_experiments(
        n_experiments=10,
        n_iter=15,
        warm_up_runs=3,
        save_dir="./results/doe/",
    )

    print(f"Completed {len(sobol_exps)} Sobol experiments and {len(lookahead_exps)} Look-ahead experiments")

# %% [markdown]
### Code for figuring out how many iterations to run for the DOE, in order to get a estimate with a given uncertainty


# %%


def run_trials_until_convergence(
    experiment: Experiment,
    warm_up_generator: Callable[[Experiment], GeneratorRun],
    doe_generator: Callable[[Experiment], GeneratorRun],
    reference_qoi_mean: float,
    sem_threshold: float = 0.01,
    error_tolerance: float = 0.01,
    warm_up_runs: int = 8,
) -> dict[str, Any]:
    """Run DOE trials until Standard Error of Mean(SEM) and QoI error is reduced to given thresholds.

    Args:
        experiment: Experiment to perform DOE on
        warm_up_generator: Generator for initial training data
        doe_generator: Generator for DOE iterations
        reference_qoi_mean: Reference mean value for error calculation
        sem_threshold: uncertainty threshold for standard error of mean (SEM)
        error_tolerance: Relative error tolerance of the QoI Mean for stopping
        warm_up_runs: Number of warm-up runs before DOE

    Returns:
        dict: Results including convergence info and final metrics
    """
    for _ in range(warm_up_runs):
        generator_run = warm_up_generator(experiment)
        trial = experiment.new_trial(generator_run)
        _ = trial.run()
        _ = trial.mark_completed()

    converged = False
    doe_iteration = 0
    convergence_info = {
        "uncertainty_converged_at": None,
        "error_converged_at": None,
        "both_converged_at": None,
        "stopped_early": False,
    }
    check_interval = 1

    while not converged:
        # Perform DOE iteration
        generator_run = doe_generator(experiment)
        trial = experiment.new_trial(generator_run=generator_run)
        _ = trial.run()
        _ = trial.mark_completed()
        doe_iteration += 1

        # Check stopping criteria at intervals and after minimum iterations
        if doe_iteration % check_interval == 0:
            # Get current metrics
            metrics = experiment.fetch_data()
            qoi_metrics = metrics.df[metrics.df["metric_name"] == "QoIMetric"]

            if len(qoi_metrics) > 0:
                # Get the latest QoI estimate
                latest_metrics = qoi_metrics.iloc[-1]
                current_mean = latest_metrics["mean"]
                current_sem = latest_metrics["sem"]
                current_trial = latest_metrics["trial_index"]

                # Check uncertainty criterion
                uncertainty_met = current_sem <= sem_threshold
                if uncertainty_met and convergence_info["uncertainty_converged_at"] is None:
                    convergence_info["uncertainty_converged_at"] = current_trial

                # Check error criterion
                relative_error = abs(current_mean - reference_qoi_mean) / abs(reference_qoi_mean)
                error_met = relative_error <= error_tolerance
                if error_met and convergence_info["error_converged_at"] is None:
                    convergence_info["error_converged_at"] = current_trial

                # Check if both criteria are met
                both_met = uncertainty_met and error_met
                if both_met and convergence_info["both_converged_at"] is None:
                    convergence_info["both_converged_at"] = current_trial
                    converged = True
                    convergence_info["stopped_early"] = True
                    break

    return {
        "total_doe_iterations": doe_iteration,
        "total_trials": warm_up_runs + doe_iteration,
        "convergence_info": convergence_info,
        "converged": converged,
    }


def run_single_convergence_experiment(
    sem_threshold: float,
    error_tolerance: float = 0.05,
    warm_up_runs: int = 8,
    seed: int = 42,
) -> tuple[Experiment, Experiment, dict[str, Any], dict[str, Any]]:
    """Run Sobol and Look-ahead experiments until stopping criteria are met.

    Args:
        sem_threshold: uncertainty threshold
        error_tolerance: Relative error tolerance from brute force QoI
        warm_up_runs: Number of warm-up runs
        seed: Random seed for reproducibility

    Returns:
        sobol_experiment: Sobol experiment object
        lookahead_experiment: Look-ahead experiment object
        sobol_results: Results from Sobol experiment
        lookahead_results: Results from Look-ahead experiment

    """
    exp_sobol = make_exp()
    _ = exp_sobol.add_tracking_metric(QOI_METRIC)
    sobol = Models.SOBOL(search_space=exp_sobol.search_space, seed=seed)
    sobol_generator_run = create_sobol_generator(sobol)

    # Run Sobol experiment until convergence
    sobol_results = run_trials_until_convergence(
        experiment=exp_sobol,
        warm_up_generator=sobol_generator_run,
        doe_generator=sobol_generator_run,
        reference_qoi_mean=brute_force_qoi,
        sem_threshold=sem_threshold,
        error_tolerance=error_tolerance,
        warm_up_runs=warm_up_runs,
    )

    exp_lookahead = make_exp()
    _ = exp_lookahead.add_tracking_metric(QOI_METRIC)
    sobol_la = Models.SOBOL(search_space=exp_lookahead.search_space, seed=seed)

    # Run Look-ahead experiment until convergence
    lookahead_results = run_trials_until_convergence(
        experiment=exp_lookahead,
        warm_up_generator=create_sobol_generator(sobol_la),
        doe_generator=look_ahead_generator_run,
        reference_qoi_mean=brute_force_qoi,
        sem_threshold=sem_threshold,
        error_tolerance=error_tolerance,
        warm_up_runs=warm_up_runs,
    )

    return exp_sobol, exp_lookahead, sobol_results, lookahead_results


def plot_qoi_convergence(
    exp_sobol: Experiment,
    exp_lookahead: Experiment,
    sem_threshold: float,
    error_tolerance: float,
    save_path: str | None = None,
    *,
    show_n_sim_runs: bool = True,
) -> None:
    """Plot QoI convergence comparison between Sobol and look-ahead experiments.

    Args:
        exp_sobol: Sobol experiment
        exp_lookahead: Look-ahead experiment
        sem_threshold: Uncertainty threshold used
        error_tolerance: Error tolerance used
        save_path: Path to save the plot
        show_n_sim_runs: Whether to show number of simulation runs on x-axis or trial number
    """
    # Get data for both experiments
    sobol_metrics = exp_sobol.fetch_data()
    sobol_qoi = sobol_metrics.df[sobol_metrics.df["metric_name"] == "QoIMetric"]
    lookahead_metrics = exp_lookahead.fetch_data()
    lookahead_qoi = lookahead_metrics.df[lookahead_metrics.df["metric_name"] == "QoIMetric"]

    # Calculate the x-axis range for fill_between
    max_trials = max(
        sobol_qoi["trial_index"].max() if len(sobol_qoi) > 0 else 0,
        lookahead_qoi["trial_index"].max() if len(lookahead_qoi) > 0 else 0,
    )

    marker = "-" if show_n_sim_runs else "o-"
    fig1, ax1 = plt.subplots(1, 1, figsize=(12, 8))

    sobol_valid = sobol_qoi.dropna(subset=["mean", "sem"])
    _ = ax1.plot(
        sobol_valid["trial_index"],
        sobol_valid["mean"],
        marker,
        color="blue",
        label="Sobol",
        markersize=4,
        linewidth=2,
        alpha=0.8,
    )

    confidence_interval = 1.96
    upper_bound = sobol_valid["mean"] + confidence_interval * sobol_valid["sem"]
    lower_bound = sobol_valid["mean"] - confidence_interval * sobol_valid["sem"]

    _ = ax1.fill_between(
        sobol_valid["trial_index"],
        lower_bound,
        upper_bound,
        alpha=0.2,
        color="blue",
        label="Sobol 95% CI",
    )

    lookahead_valid = lookahead_qoi.dropna(subset=["mean", "sem"])
    _ = ax1.plot(
        lookahead_valid["trial_index"],
        lookahead_valid["mean"],
        marker,
        color="green",
        label="Look-ahead DOE",
        markersize=4,
        linewidth=2,
        alpha=0.8,
    )

    confidence_interval = 1.96
    upper_bound = lookahead_valid["mean"] + confidence_interval * lookahead_valid["sem"]
    lower_bound = lookahead_valid["mean"] - confidence_interval * lookahead_valid["sem"]

    _ = ax1.fill_between(
        lookahead_valid["trial_index"],
        lower_bound,
        upper_bound,
        alpha=0.2,
        color="green",
        label="Look-ahead 95% CI",
    )

    # Add reference line
    _ = ax1.axhline(brute_force_qoi, c="black", label="Brute force QoI", linestyle="--", linewidth=2)

    # Add error tolerance bands (inner, darker red)
    error_band = abs(brute_force_qoi) * error_tolerance
    _ = ax1.fill_between(
        [0, max_trials],
        brute_force_qoi - error_band,
        brute_force_qoi + error_band,
        alpha=0.3,
        color="red",
        label=f"Error tolerance (±{error_tolerance * 100:.1f}%)",
    )

    uncertainty_band = sem_threshold * 1.96  # 95% confidence interval
    _ = ax1.fill_between(
        [0, max_trials],
        brute_force_qoi - uncertainty_band,
        brute_force_qoi + uncertainty_band,
        alpha=0.15,
        color="red",
        label=f"SEM threshold (±{sem_threshold:.4f} SEM, 95% CI)",
    )

    _ = ax1.axhline(brute_force_qoi + error_band, c="red", alpha=0.5, linestyle=":")
    _ = ax1.axhline(brute_force_qoi - error_band, c="red", alpha=0.5, linestyle=":")
    _ = ax1.axhline(brute_force_qoi + uncertainty_band, c="red", alpha=0.3, linestyle="-.")
    _ = ax1.axhline(brute_force_qoi - uncertainty_band, c="red", alpha=0.3, linestyle="-.")

    stop_trial = sobol_qoi["trial_index"].max()
    _ = ax1.axvline(
        stop_trial,
        color="blue",
        linestyle="-.",
        alpha=0.8,
        linewidth=2,
        label=f"Sobol stopped (trial {stop_trial})",
    )

    stop_trial = lookahead_qoi["trial_index"].max()
    _ = ax1.axvline(
        stop_trial,
        color="green",
        linestyle="-.",
        alpha=0.8,
        linewidth=2,
        label=f"Look-ahead stopped (trial {stop_trial})",
    )

    if show_n_sim_runs:
        current_ticks = ax1.get_xticks()
        new_labels = [str(int(tick * N_SIMULATIONS_PER_POINT)) if tick >= 0 else "" for tick in current_ticks]
        _ = ax1.set_xticks(current_ticks)
        _ = ax1.set_xticklabels(new_labels)

        _ = ax1.set_xlabel("Number of Simulation Runs")
    else:
        _ = ax1.set_xlabel("Trial Number")
    _ = ax1.set_ylabel("QoI Estimate")
    _ = ax1.set_title("QoI Convergence with 95% Confidence bound: Sobol vs Look-ahead DOE")
    _ = ax1.legend()
    ax1.grid(visible=True, alpha=0.3)

    if save_path:
        qoi_save_path = f"{save_path}qoi_convergence.png"
        plt.savefig(qoi_save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_sem_convergence(
    exp_sobol: Experiment,
    exp_lookahead: Experiment,
    sobol_results: dict[str, Any],
    lookahead_results: dict[str, Any],
    sem_threshold: float,
    save_path: str | None = None,
    *,
    show_n_sim_runs: bool = True,
) -> None:
    """Plot SEM convergence comparison between Sobol and look-ahead experiments.

    Args:
        exp_sobol: Sobol experiment
        exp_lookahead: Look-ahead experiment
        sobol_results: Sobol convergence results
        lookahead_results: Look-ahead convergence results
        sem_threshold: Uncertainty threshold used
        save_path: Path to save the plot
        show_n_sim_runs: Whether to show number of simulation runs on x-axis or trial number
    """
    # Get data for both experiments
    sobol_metrics = exp_sobol.fetch_data()
    sobol_qoi = sobol_metrics.df[sobol_metrics.df["metric_name"] == "QoIMetric"]
    lookahead_metrics = exp_lookahead.fetch_data()
    lookahead_qoi = lookahead_metrics.df[lookahead_metrics.df["metric_name"] == "QoIMetric"]

    marker = "-" if show_n_sim_runs else "o-"
    fig2, ax2 = plt.subplots(1, 1, figsize=(12, 8))

    if len(sobol_qoi) > 0:
        sobol_valid = sobol_qoi.dropna(subset=["sem"])
        if len(sobol_valid) > 0:
            _ = ax2.plot(
                sobol_valid["trial_index"],
                sobol_valid["sem"],
                marker,
                color="blue",
                label="Sobol SEM",
                markersize=6,
                linewidth=2,
                markerfacecolor="lightblue",
                markeredgecolor="blue",
                markeredgewidth=1,
            )

    if len(lookahead_qoi) > 0:
        lookahead_valid = lookahead_qoi.dropna(subset=["sem"])
        if len(lookahead_valid) > 0:
            _ = ax2.plot(
                lookahead_valid["trial_index"],
                lookahead_valid["sem"],
                marker,
                color="green",
                label="Look-ahead SEM",
                markersize=6,
                linewidth=2,
                markerfacecolor="lightgreen",
                markeredgecolor="green",
                markeredgewidth=1,
            )

    # Add SEM threshold line
    _ = ax2.axhline(
        sem_threshold,
        color="red",
        linestyle="--",
        alpha=0.8,
        linewidth=3,
        label=f"SEM threshold ({sem_threshold:.4f})",
    )

    if sobol_results["converged"]:
        stop_trial = sobol_results["total_trials"] - 1
        _ = ax2.axvline(stop_trial, color="blue", linestyle="-.", alpha=0.7, linewidth=2)

    if lookahead_results["converged"]:
        stop_trial = lookahead_results["total_trials"] - 1
        _ = ax2.axvline(stop_trial, color="green", linestyle="-.", alpha=0.7, linewidth=2)

    if show_n_sim_runs:
        current_ticks = ax2.get_xticks()
        new_labels = [str(int(tick * N_SIMULATIONS_PER_POINT)) if tick >= 0 else "" for tick in current_ticks]
        _ = ax2.set_xticks(current_ticks)
        _ = ax2.set_xticklabels(new_labels)

        _ = ax2.set_xlabel("Number of Simulation Runs")
    else:
        _ = ax2.set_xlabel("Trial Number")
    _ = ax2.set_ylabel("Standard Error of Mean (SEM)")
    _ = ax2.set_title("Standard Error of Mean Reduction Comparison (Individual SEM Values)")
    _ = ax2.legend()
    ax2.grid(visible=True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        sem_save_path = f"{save_path}sem_convergence.png"
        plt.savefig(sem_save_path, dpi=300, bbox_inches="tight")

    plt.show()


# %%
# Example usage: Run convergence experiment with specified criteria
if __name__ == "__main__":
    # Set your desired stopping criteria
    sem_percentage = 0.01  # 1% of brute force QoI as sem threshold
    sem_threshold = abs(brute_force_qoi) * sem_percentage
    error_tolerance = 0.005  # 0.5% relative error from brute force QoI

    print(f"- SEM threshold: {sem_threshold:.4f} ({sem_percentage * 100:.1f}% of brute force QoI)")
    print(f"- Error tolerance: {error_tolerance * 100:.1f}%")
    print(f"- Brute force QoI reference: {brute_force_qoi:.4f}")

    # Run the convergence experiment with dynamic stopping
    exp_sobol, exp_lookahead, sobol_results, lookahead_results = run_single_convergence_experiment(
        sem_threshold=sem_threshold,
        error_tolerance=error_tolerance,
        warm_up_runs=8,
        seed=42,
    )

    # %%
    plot_qoi_convergence(
        exp_sobol=exp_sobol,
        exp_lookahead=exp_lookahead,
        sem_threshold=sem_threshold,
        error_tolerance=error_tolerance,
        save_path="./results/doe/",
    )

    plot_sem_convergence(
        exp_sobol=exp_sobol,
        exp_lookahead=exp_lookahead,
        sobol_results=sobol_results,
        lookahead_results=lookahead_results,
        sem_threshold=sem_threshold,
        save_path="./results/doe/",
    )


# %%
