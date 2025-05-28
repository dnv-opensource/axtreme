"""The script explores the sensitivity of DOE results to the sobol seed.

The sobol seed controls the initial dataset generated (used to train the GP),
and the baseline performance the DOE/acquisition function is compared to.
This runs using the QoI and Experiment define in doe.py and is intend as
an additional analysis of how the sobol seed effects the DOE process defined there.
"""

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ax import Experiment
from ax.modelbridge.registry import Models
from doe import (  # type: ignore[import-not-found]
    QOI_METRIC,
    brute_force_qoi_estimate,
    create_sobol_generator,
    look_ahead_generator_run,
    make_exp,
    plot_qoi_estimates_from_experiment,
    run_trials,
)

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

        run_trials(
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

        run_trials(
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
        brute_force_qoi_estimate=brute_force_qoi_estimate,
        save_dir=save_dir,
    )

    # Create summary plot
    print("Creating summary plot...")
    create_summary_plot(
        sobol_experiments,
        look_ahead_experiments,
        n_iter,
        brute_force_qoi_estimate,
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
    """Create comparison plots for multiple experiments.

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

# %%
