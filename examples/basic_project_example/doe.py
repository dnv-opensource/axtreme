"""Design of Experiments (DOE) analysis.

This module demonstrates the application of the axtreme library for performing
Design of Experiments. It uses

Look-ahead DOE: axtreme acquisition function that picks the next point based on
the expected uncertainty reduction in the QoI estimate of the GP.

The analysis includes:
- Visualization of acquisition function surfaces
- GP model fitting and validation
"""

# %%
import json
from collections.abc import Callable
from pathlib import Path
from time import gmtime, strftime

import cmcrameri.cm as cmc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from ax import Experiment
from ax.core import GeneratorRun
from ax.modelbridge import ModelBridge
from ax.modelbridge.registry import Models
from cmap import Colormap
from IPython.display import display
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from problem import QOI_ESTIMATOR, brute_force_qoi, make_exp  # type: ignore[import-not-found]

from axtreme import experiment, sampling
from axtreme.acquisition import QoILookAhead
from axtreme.metrics import QoIMetric
from axtreme.plotting.doe import plot_qoi_estimates_from_experiment
from axtreme.plotting.gp_fit import plot_gp_fits_2d_surface_from_experiment
from axtreme.utils import transforms

torch.set_default_dtype(torch.float64)
device = "cpu"

# pyright: reportUnnecessaryTypeIgnoreComment=false

# %% Global variables
STOPPING_CRITERIA_PERCENTAGE = 5  # Percentage for relative SEM stopping criteria
USE_EXISTING_SIMULATION_DATA = True  # Whether to use existing simulation data or run new simulations
USE_EXISTING_QOI_DATA = False  # Whether to use existing QoI data or start DoE from scratch

# %% [markdown]
# ## Set up helper functions for running the DOE


def save_log_current_trial(exp: Experiment, log_file_path: str) -> None:
    """Save the current trial data to a log file in JSON format."""
    # Extract all data from the experiment
    exp_data = experiment.extract_data_from_experiment_as_json(exp)

    # Load existing log or initialize empty dict
    log_path = Path(log_file_path)
    if log_path.exists():
        with log_path.open() as f:
            data = json.load(f)
    else:
        data = {}

    missing_keys = set(exp_data.keys()) - {int(k) for k in data}

    for key in missing_keys:
        data[key] = exp_data[key]

    # Save updated data back to file
    with log_path.open() as f:
        json.dump(data, f, indent=4)


def run_trials(
    experiment: Experiment,
    warm_up_generator: Callable[[Experiment], GeneratorRun],
    doe_generator: Callable[[Experiment], GeneratorRun],
    warm_up_runs: int = 3,
    doe_runs: int = 15,
    stopping_criteria: Callable[[Experiment], bool] | None = None,
    log_file_path: str | None = None,
) -> int:
    """Run trials for an experiment and return the QOI results using QoI metric.

    Args:
        experiment: Experiment to perform DOE on.
        warm_up_generator: Generator to create the initial training data on the experiment (e.g., Sobol).
        doe_generator: The generator being used to perform the DoE.
        warm_up_runs: Number of warm-up runs to perform before starting the DoE.
        doe_runs: Number of DoE runs to perform.
        stopping_criteria: Optional function that takes an experiment and returns True if a given
        stopping criteria is met. If this stopping criteria is not met after `doe_runs` iterations, the function will
            return the number of iterations run.
        log_file_path: Optional path to save the log file after each trial. If None, a default path with timestamp is
            used.

    """
    # Save log in a file with timestamp (UTC time)
    if log_file_path is None:
        log_file_path = f"results/doe_log_{strftime('%Y_%m_%d_%H_%M', gmtime())}.json"

    for i in range(doe_runs + 1):
        # Warm-up phase
        if (i == 0) & (warm_up_runs > 0):
            for _ in range(warm_up_runs):
                generator_run = warm_up_generator(experiment)
                trial = experiment.new_trial(generator_run)
                _ = trial.run()
                _ = trial.mark_completed()

                # Save progress after each DoE iteration
                save_log_current_trial(experiment, log_file_path)

        else:
            generator_run = doe_generator(experiment)
            trial = experiment.new_trial(generator_run=generator_run)
            _ = trial.run()
            _ = trial.mark_completed()

            # Save progress after each DoE iteration
            save_log_current_trial(experiment, log_file_path)

        print(f"iter {i} done")

        # Check stopping criteria after each DoE iteration
        if stopping_criteria is not None and stopping_criteria(experiment):
            print(f"Stopping criteria met after {i} DoE iterations")
            return i

    return doe_runs + warm_up_runs


def _latest_qoi(experiment: Experiment, metric_name: str) -> None | pd.DataFrame:
    """Get the latest QoI metric row from the experiment data."""
    df = experiment.fetch_data().df
    qoi = df[df["metric_name"] == metric_name]
    if qoi.empty:
        print(f"[StoppingCriteria] No '{metric_name}' data found.")
        return None
    row = qoi.iloc[-1]
    if pd.isna(row.get("sem")) or pd.isna(row.get("mean")):
        print(f"[StoppingCriteria] Missing SEM or mean for '{metric_name}'.")
        return None
    return row


def sem_stopping_criteria_absolute(
    experiment: Experiment, sem_threshold: float = 0.1, metric_name: str = "QoIMetric"
) -> bool:
    """Stop if latest SEM ≤ fixed threshold."""
    row = _latest_qoi(experiment, metric_name)
    if row is None:
        return False
    sem = row["sem"]
    ok = sem <= sem_threshold
    print(f"[StoppingCriteria] {'✅' if ok else '❌'} SEM={sem:.4f}, threshold={sem_threshold}")
    return ok


def sem_stopping_criteria_relative(
    experiment: Experiment, mean_percentage: float = STOPPING_CRITERIA_PERCENTAGE, metric_name: str = "QoIMetric"
) -> bool:
    """Stop if latest SEM ≤ |mean| * mean_percentage / 100."""
    row = _latest_qoi(experiment, metric_name)
    if row is None:
        return False
    sem, mean = row["sem"], row["mean"]
    threshold = abs(mean) * (mean_percentage / 100)
    ok = sem <= threshold
    print(
        f"[StoppingCriteria] {'✅' if ok else '❌'} SEM={sem:.4f}, threshold={threshold:.4f} "
        f"(aim: {mean_percentage}% of |mean|)"
    )
    return ok


# This is used to create random Sobol points for the warm-up phase.
def create_sobol_generator(sobol: ModelBridge) -> Callable[[Experiment], GeneratorRun]:
    """Closure helper to run a sobol generator in the interface run_trails required.

    Note the typing is a bit general -> should be a sobol generator.

    Returns:
        Callable[[Experiment], GeneratorRun]: A function that takes an experiment and returns a generator run.
    """

    def sobol_generator_run(_: Experiment) -> GeneratorRun:
        return sobol.gen(1)

    return sobol_generator_run


def look_ahead_generator_run(experiment: Experiment) -> GeneratorRun:
    """Generate the next optimal point using QoI look-ahead acquisition function.

    Builds a GP model from current experiment data and uses QoILookAhead acquisition
    function to select the next point that reduces QoI uncertainty the most.

    Args:
        experiment: Ax experiment containing current trial data.

    Returns:
        GeneratorRun containing the next optimal point to evaluate.

    Note:
        Uses two-step model building to handle transform issues with QoI metrics.
        Refits GP hyperparameters on each call.
    """
    # Fist building model to get the transforms
    model_bridge_only_model = Models.BOTORCH_MODULAR(
        experiment=experiment,
        data=experiment.fetch_data(metrics=list(experiment.optimization_config.metrics.values())),  # type: ignore  # noqa: PGH003
        fit_tracking_metrics=False,  # Needed for QoIMetric to work properly
    )
    input_transform, outcome_transform = transforms.ax_to_botorch_transform_input_output(
        transforms=list(model_bridge_only_model.transforms.values()), outcome_names=model_bridge_only_model.outcomes
    )
    # Adding the transforms to the QoI estimator
    QOI_ESTIMATOR.input_transform = input_transform
    QOI_ESTIMATOR.outcome_transform = outcome_transform

    # Building the model with the QoILookAhead acquisition function
    model_bridge_cust_ac = Models.BOTORCH_MODULAR(
        experiment=experiment,
        data=experiment.fetch_data(),
        botorch_acqf_class=QoILookAhead,
        fit_tracking_metrics=False,  # Needed for QoIMetric to work properly
        acquisition_options={
            "qoi_estimator": QOI_ESTIMATOR,
            "sampler": sampling.MeanSampler(),
        },
    )

    # Optimizing the acquisition function to get the next point
    return model_bridge_cust_ac.gen(
        1,
        # Note these arg are supplied by default for this method.
        model_gen_options={
            "optimizer_kwargs": {
                "num_restarts": 20,
                "raw_samples": 50,
                "options": {
                    "with_grad": False,
                    "method": "Nelder-Mead",
                    "maxfev": 5,
                },
                "retry_on_optimization_warning": False,
            }
        },
    )


# %% [markdown]
# ## DoE (i.e. procedure to select next points for which simulations are run)

# How many iterations to run in the following DOEs
n_iter_doe = 30
# The minimum number of data points the experiment must have before the GP is trained and the QoI is actually run
warm_up_runs = 5
# Set the number of warm-up runs for the DOE process
doe_warm_up_runs = warm_up_runs

# Define the QoI metric to track during the experiment
QOI_METRIC = QoIMetric(
    name="QoIMetric", qoi_estimator=QOI_ESTIMATOR, minimum_data_points=warm_up_runs, attach_transforms=True
)

# Make experiment. In this variable all relevant information about the problem is stored.
exp_look_ahead = make_exp()

# %% [markdown]
# ## Optionally load existing data into the experiment either from existing simulations or previous DoE runs
# Warning: if the choice of existing data is poor (i.e. not covering the input space well) the GP model may be
# poorly fitted and the DoE process may not converge well.

# Optional (Option 1): Add existing simulation and env data.
if USE_EXISTING_SIMULATION_DATA:
    sim_log_file_path = "results/simulation_examples.npz"  # Update with the actual log file path
    loaded_data = np.load(sim_log_file_path)
    env_data_loaded = loaded_data["env_data"].astype(
        float
    )  # Important to cast to float as ax does not accept np.float32
    simulator_samples_loaded = loaded_data["simulator_samples"].astype(float)

    # Load the simulation data into the experiment
    # Points need to be loaded individually to create a separate trial for each point.
    for env_point, sim_outputs in zip(env_data_loaded, simulator_samples_loaded, strict=False):
        _, _ = experiment.add_simulation_data_to_experiment(
            experiment=exp_look_ahead,
            parameters=["x1", "x2"],
            simulation_inputs=[env_point],
            simulation_outputs=[sim_outputs],
        )

    # If existing data is loaded and includes at least warm_up_runs runs, warm-up runs should be set to 0
    doe_warm_up_runs = 0 if env_data_loaded.shape[0] >= doe_warm_up_runs else warm_up_runs - env_data_loaded.shape[0]

# Add the QoI metric to the experiment
# Important: needs to be added after loading existing simulation data as no QoI info is available yet.
# But needs to be added before loading existing QoI runs.
_ = exp_look_ahead.add_tracking_metric(QOI_METRIC)

# Optional (Option 2): Add previous DoE runs, i.e. incl. QoI for each iteration
if USE_EXISTING_QOI_DATA:
    with Path("results/doe_log_2025_11_26_12_30.json").open() as file:
        doe_log = json.load(file)

    experiment.add_json_data_to_experiment(experiment=exp_look_ahead, json_data=doe_log)

    # If existing data is loaded and includes at least doe_warm_up_runs runs, doe_warm_up_runs runs should be set to 0
    doe_warm_up_runs = 0 if len(doe_log) >= doe_warm_up_runs else warm_up_runs - len(doe_log)

# %%

# This needs to be instantiated outside of the loop so the internal state of the generator persists.
sobol = Models.SOBOL(search_space=exp_look_ahead.search_space, seed=5)

# Run the DoE
last_itr_look_ahead = run_trials(
    experiment=exp_look_ahead,
    warm_up_generator=create_sobol_generator(sobol),  # Can be none when using existing data for warm-up
    doe_generator=look_ahead_generator_run,
    warm_up_runs=doe_warm_up_runs,
    doe_runs=n_iter_doe,
    stopping_criteria=sem_stopping_criteria_relative,  # Optional: use a stopping criteria based on confidence bound
)

# Show the QoI results for all iterations
experiment_df = exp_look_ahead.fetch_data().df
display(experiment_df)

# %% [markdown]
# ## Plot the QoI estimates over the DOE iterations

_, ax = plt.subplots()
ax = plot_qoi_estimates_from_experiment(exp_look_ahead, ax=ax, color="green", name="look ahead")
_ = ax.axhline(brute_force_qoi, c="black", label="brute_force_value")
_ = ax.set_xlabel("Number of DOE iterations")
_ = ax.set_ylabel("Response")
_ = ax.legend()

# %% Plot the loc and scale surface estimated for the warm up phase and after the last iteration of the DoE process
# Warm up runs can also be changed to the length of the existing data
fig_trial_warm_up = plot_gp_fits_2d_surface_from_experiment(exp_look_ahead, warm_up_runs)
fig_trial_warm_up.show()
fig_last_trial = plot_gp_fits_2d_surface_from_experiment(exp_look_ahead, experiment_df["trial_index"].max() + 1)
fig_last_trial.show()


# %% [markdown]
# ## Plot the points selected during the DoE process
def plot_2dtrials(
    exp: Experiment,
    ax: Axes,
    marker: str = "o",
    cmap: Colormap = cmc.batlow_r,
    *,
    show_colorbar: bool = True,
) -> Axes:
    """Plot 2D trial points with order-colored markers using a Crameri colormap."""
    # Collect valid trial points and indices
    trials = [(list(t.arm.parameters.values()), i) for i, t in exp.trials.items() if t.arm]

    points, trial_indices = map(np.array, zip(*trials, strict=False))
    n = len(points)
    norm = Normalize(0, n - 1)
    colors = cmap(norm(range(n)))

    # Plot points and labels
    for (x, y), c, idx in zip(points, colors, trial_indices, strict=False):
        _ = ax.scatter(x, y, color=c, marker=marker, s=50, alpha=0.9)
        _ = ax.annotate(str(idx), (x, y), xytext=(2, 2), textcoords="offset points", fontsize=8, color=c)

    if show_colorbar:
        sm = ScalarMappable(norm=norm, cmap=cmap)
        plt.colorbar(sm, ax=ax, pad=0.02, fraction=0.046).set_label("Trial Order", rotation=270, labelpad=15)

    return ax


fig, ax = plt.subplots(figsize=(5, 5))
_ = ax.set(xlabel="x1", ylabel="x2")
_ = plot_2dtrials(exp_look_ahead, ax=ax, marker="^", show_colorbar=True)
ax.grid()
_ = ax.legend()
plt.tight_layout()
plt.show()

# %%
