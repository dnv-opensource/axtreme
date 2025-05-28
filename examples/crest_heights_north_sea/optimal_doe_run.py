"""Code for running only the new `ExperimentBasedAcquisition` optimal acquisition function fomr the optimal_doe.py file.

It is for now working until the surface plot.
"""

# %%
from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from ax import (
    Experiment,
)
from ax.core import GeneratorRun
from ax.modelbridge import ModelBridge
from ax.modelbridge.registry import Models
from numpy.typing import NDArray
from problem import (  # type: ignore[import-not-found]
    DIST,
    make_exp,
    period_length,
)
from simulator import max_crest_height_simulator_function  # type: ignore[import-not-found]
from torch.utils.data import DataLoader
from usecase.env_data import collect_data  # type: ignore[import-not-found]

from axtreme.data import FixedRandomSampler, MinimalDataset
from axtreme.experiment import add_sobol_points_to_experiment
from axtreme.metrics import QoIMetric
from axtreme.plotting.gp_fit import plot_gp_fits_2d_surface_from_experiment
from axtreme.qoi import MarginalCDFExtrapolation
from axtreme.sampling.ut_sampler import UTSampler
from axtreme.utils import transforms

torch.set_default_dtype(torch.float64)
device = "cpu"

# pyright: reportUnnecessaryTypeIgnoreComment=false

# %%


dist = DIST


# %%
raw_data: pd.DataFrame = collect_data()

# we convert this data to Numpy for ease of use from here on out
env_data: NDArray[np.float64] = raw_data.to_numpy()

# %%
# Bruteforce estimate placeholder
# TODO:(@henrikstoklandberg 2025-04-26): This is a placholder(fast and ugly implementation) for the brute force estimate of the QOI.
# This should be repalace by a bruteforce GP fit running a lot of sobol points and then using this as the baseline for comparison.
# Q: How many iteratins of the DOE is needed to get the same estimate as 1000 sobol points?
n_erd_samples = 1000
erd_samples = []
for _ in range(n_erd_samples):
    indices = np.random.choice(env_data.shape[0], size=period_length, replace=True)  # noqa: NPY002
    period_sample = env_data[indices]

    responses = max_crest_height_simulator_function(period_sample)
    erd_samples.append(responses.max())

brute_force_qoi_estimate = np.median(erd_samples)
print(f"Brute force estimate of our QOI is {brute_force_qoi_estimate}")


# %%
def run_trials(
    experiment: Experiment,
    warm_up_generator: Callable[[Experiment], GeneratorRun],
    doe_generator: Callable[[Experiment], GeneratorRun],
    warm_up_runs: int = 3,
    doe_runs: int = 15,
) -> None:
    """Helper function for running  trials for an experiment and returning the QOI results using QoI metric.

    Args:
        experiment: Experiment to perform DOE on.
        warm_up_generator: Generator to create the initial training data on the experiment (e.g., Sobol).
        doe_generator: The generator being used to perform the DoE.
        warm_up_runs: Number of warm-up runs to perform before starting the DoE.
        doe_runs: Number of DoE runs to perform.

    """
    for i in range(doe_runs + 1):
        if i == 0:
            for _ in range(warm_up_runs):
                generator_run = warm_up_generator(experiment)
                trial = experiment.new_trial(generator_run)
                _ = trial.run()
                _ = trial.mark_completed()

        else:
            generator_run = doe_generator(experiment)
            trial = experiment.new_trial(generator_run=generator_run)
            _ = trial.run()
            _ = trial.mark_completed()
        print(f"iter {i} done")


# %%
# QOI estimator
# A note on DataLoaders:
# We make use of Dataloader to manage providing data to the QoI. Env samples are only used in the QoI, so your env data
# can be in whatever format is supported by the QoI.
n_env_samples = 1_000
dataset = MinimalDataset(env_data)
sampler = FixedRandomSampler(dataset, num_samples=n_env_samples, seed=10, replacement=True)
dataloader = DataLoader(dataset, sampler=sampler, batch_size=256)

posterior_sampler = UTSampler()
# posterior_sampler = NormalIndependentSampler(torch.Size([n_posterior_samples])

QOI_ESTIMATOR = MarginalCDFExtrapolation(
    # random dataloader give different env samples for each instance
    env_iterable=dataloader,
    period_len=period_length,
    quantile=torch.tensor(0.5),
    quantile_accuracy=torch.tensor(0.01),
    # IndexSampler needs to be used with GenericDeterministicModel. Each sample just selects the mean.
    posterior_sampler=posterior_sampler,
)


# %% [markdown]
# How many iterations to run in the following DOEs

# %%
n_iter = 40  # 15  # 40
warm_up_runs = 6  # 3

# %% [markdown]
# ### Sobol model
# Surrogate trained without and a acquisition function as a comparative baseline.

# %%
# Create a constant QoI tracking metric used for the following experiments.
QOI_METRIC = QoIMetric(
    name="QoIMetric", qoi_estimator=QOI_ESTIMATOR, minimum_data_points=warm_up_runs, attach_transforms=True
)

# %%
exp_sobol = make_exp()

# Add the QoI metric to the experiment
_ = exp_sobol.add_tracking_metric(QOI_METRIC)

# This needs to be instantiated outside of the loop so the internal state of the generator persists.
sobol = Models.SOBOL(search_space=exp_sobol.search_space, seed=15)  # 5)


def create_sobol_generator(sobol: ModelBridge) -> Callable[[Experiment], GeneratorRun]:
    """Closure helper to run a sobol generator in the interface run_trails required.

    Note the typing is a bit general -> should be a sobol generatror.

    Returns:
        Callable[[Experiment], GeneratorRun]: A function that takes an experiment and returns a generator run.
    """

    def sobol_generator_run(_: Experiment) -> GeneratorRun:
        return sobol.gen(1)

    return sobol_generator_run


# %% [markdown]
# DEBUG todo:
# - Instance it
# - Run with a single x
# - surface plot -> Came here as of 2025-05-28
# - Single DOE run
# - Full DOE run

# %%
from importlib import reload

import optimal_doe
from optimal_doe import ExperimentBasedAcquisition  # noqa: E402

reload(optimal_doe)

# %%
test_exp = make_exp()
add_sobol_points_to_experiment(test_exp, n_iter=64, seed=8)
model_bridge_only_model = Models.BOTORCH_MODULAR(
    experiment=test_exp,
    data=test_exp.fetch_data(metrics=list(test_exp.optimization_config.metrics.values())),  # type: ignore  # noqa: PGH003
    fit_tracking_metrics=False,  # Needed for QoIMetric to work properly
)
input_transform, outcome_transform = transforms.ax_to_botorch_transform_input_output(
    transforms=list(model_bridge_only_model.transforms.values()), outcome_names=model_bridge_only_model.outcomes
)
QOI_ESTIMATOR = MarginalCDFExtrapolation(
    # random dataloader give different env samples for each instance
    env_iterable=dataloader,
    period_len=period_length,
    quantile=torch.tensor(0.5),
    quantile_accuracy=torch.tensor(0.01),
    # IndexSampler needs to be used with GenericDeterministicModel. Each sample just selects the mean.
    posterior_sampler=posterior_sampler,
)


# %%
acqf = ExperimentBasedAcquisition(
    model_bridge_only_model.model.surrogate.model, QOI_ESTIMATOR, test_exp, input_transform
)


# %%
model_space_tensor = torch.tensor([[[0.5, 0.5]]])
# model_space_tensor = torch.tensor([[[0.8, 0.5]]])
point_acqf_val = acqf(model_space_tensor)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
print("point_acqf_val", point_acqf_val)


# %%
import copy

from ax.core import Arm

# Store the original experiment state before adding the point
QOI_ESTIMATOR_TEST = MarginalCDFExtrapolation(
    env_iterable=dataloader,
    period_len=period_length,
    quantile=torch.tensor(0.5),
    quantile_accuracy=torch.tensor(0.01),
    posterior_sampler=posterior_sampler,
)

# Store the original experiment state before adding the point
original_test_exp = copy.deepcopy(test_exp)

# Convert from model space to problem space and add point
search_space_parameters = test_exp.search_space.parameters
Hs = search_space_parameters["Hs"]
Tp = search_space_parameters["Tp"]

Hs_problem_space = Hs.lower + model_space_tensor[0, 0, 0] * (Hs.upper - Hs.lower)
Tp_problem_space = Tp.lower + model_space_tensor[0, 0, 1] * (Tp.upper - Tp.lower)

print("[Hs_problem_space, Tp_problem_space]", [Hs_problem_space, Tp_problem_space])
problem_space_tensor = torch.tensor([Hs_problem_space.item(), Tp_problem_space.item()])
print("problem_space_tensor", problem_space_tensor)

# %% [markdown]
### Check for on point added to the experiment
# Add point to experiment
trial = test_exp.new_trial()
trial.add_arm(
    Arm(parameters={k: v.item() for k, v in zip(test_exp.parameters.keys(), problem_space_tensor, strict=True)})
)
_ = trial.run()
_ = trial.mark_completed()

# Create acquisition function using the ORIGINAL experiment (before adding the point)
acqf = ExperimentBasedAcquisition(
    model_bridge_only_model.model.surrogate.model, QOI_ESTIMATOR_TEST, original_test_exp, input_transform
)

# Get acquisition function value
point_acqf_val = acqf(model_space_tensor)

# Create model bridge using the UPDATED experiment (after adding the point)
model_bridge = Models.BOTORCH_MODULAR(
    experiment=test_exp,
    data=test_exp.fetch_data(metrics=list(test_exp.optimization_config.metrics.values())),
    fit_tracking_metrics=False,
)

# Apply transforms and get QoI estimate
input_transform_new, outcome_transform_new = transforms.ax_to_botorch_transform_input_output(
    transforms=list(model_bridge.transforms.values()), outcome_names=model_bridge.outcomes
)

# Use the SAME QoI estimator instance for consistency
QOI_ESTIMATOR_TEST.input_transform = input_transform_new
QOI_ESTIMATOR_TEST.outcome_transform = outcome_transform_new

qoi_estimate = QOI_ESTIMATOR_TEST(model_bridge.model.surrogate.model)
qoi_estimate_var = QOI_ESTIMATOR_TEST.var(qoi_estimate)

print("qoi_estimator_var", qoi_estimate_var)
print("acqf(model_space_tensor)", point_acqf_val)


# Now they should match (with tolerance for floating point differences)
assert torch.isclose(qoi_estimate_var, point_acqf_val.squeeze(), atol=1e-3), (
    f"QOI estimate variance ({-qoi_estimate_var.item()}) does not match acquisition function value ({point_acqf_val.item()})"
)

# %%
# Check for adding point to the experiment 10 times
qoi_var_results = []
qoi_mean_results = []
acqf_var_results = []
acqf_mean_results = []
differences_mean = []
differences_var = []
n_iter = 1000  # 10
for i in range(n_iter):
    exp_itr = copy.deepcopy(test_exp)
    # Add point to experiment
    trial = exp_itr.new_trial()
    trial.add_arm(
        Arm(parameters={k: v.item() for k, v in zip(exp_itr.parameters.keys(), problem_space_tensor, strict=True)})
    )
    _ = trial.run()
    _ = trial.mark_completed()

    # Create acquisition function using the ORIGINAL experiment (before adding the point)
    acqf = ExperimentBasedAcquisition(
        model_bridge_only_model.model.surrogate.model, QOI_ESTIMATOR_TEST, original_test_exp, input_transform
    )

    # Get acquisition function value
    point_acqf_val = acqf(model_space_tensor)

    # Create model bridge using the UPDATED experiment (after adding the point)
    model_bridge = Models.BOTORCH_MODULAR(
        experiment=exp_itr,
        data=exp_itr.fetch_data(metrics=list(exp_itr.optimization_config.metrics.values())),
        fit_tracking_metrics=False,
    )

    # Apply transforms and get QoI estimate
    input_transform_new, outcome_transform_new = transforms.ax_to_botorch_transform_input_output(
        transforms=list(model_bridge.transforms.values()), outcome_names=model_bridge.outcomes
    )

    # Use the SAME QoI estimator instance for consistency
    QOI_ESTIMATOR_TEST.input_transform = input_transform_new
    QOI_ESTIMATOR_TEST.outcome_transform = outcome_transform_new

    qoi_estimate = QOI_ESTIMATOR_TEST(model_bridge.model.surrogate.model)
    qoi_estimate_var = QOI_ESTIMATOR_TEST.var(qoi_estimate)
    qoi_estimate_mean = QOI_ESTIMATOR_TEST.mean(qoi_estimate)

    print("qoi_estimator_var", qoi_estimate_var)
    print("acqf(model_space_tensor)", point_acqf_val)

    qoi_var_results.append(qoi_estimate_var.item())
    qoi_mean_results.append(qoi_estimate_mean.item())
    acqf_var_results.append(point_acqf_val.squeeze().item())

    difference_var = qoi_estimate_var.item() - point_acqf_val.squeeze().item()
    differences_var.append(difference_var)
    print("difference", difference_var)

# %%
# Convert to numpy arrays for easier plotting
qoi_var_results = np.array(qoi_var_results)
acqf_var_results = np.array(acqf_var_results)
differences_var = np.array(differences_var)

# print("qo_vari_results", qoi_var_results)
print("qoi_var_results.var", qoi_var_results.var())
print("qoi_var_results.mean", qoi_var_results.mean())
# print("acqf_var_results", acqf_var_results)
print("acqf_var_results.var", acqf_var_results.var())
print("acqf_var_results.mean", acqf_var_results.mean())
# print("differences_var", differences_var)
print("differences_var.var", differences_var.var())
print("differences_var.mean", differences_var.mean())

# %%

# Create comprehensive histogram visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Histogram of QoI variance results
axes[0, 0].hist(qoi_var_results, bins=100, alpha=0.7, color="lightblue", edgecolor="black")
axes[0, 0].hist(acqf_var_results, bins=100, alpha=0.7, color="lightgreen", edgecolor="black")
axes[0, 0].axvline(
    qoi_var_results.mean(), color="red", linestyle="--", label=f" QOI Mean: {qoi_var_results.mean():.6f}"
)
axes[0, 0].axvline(
    qoi_var_results.mean(), color="orange", linestyle="--", label=f"Acqf Mean: {qoi_var_results.mean():.6f}"
)
axes[0, 0].set_xlabel("QoI Variance Results")
axes[0, 0].set_ylabel("Frequency")
axes[0, 0].set_title("Distribution of QoI Variance vs Acqf Results")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Histogram of acquisition function results
axes[0, 1].hist(acqf_var_results, bins=8, alpha=0.7, color="lightgreen", edgecolor="black")
axes[0, 1].axvline(acqf_var_results.mean(), color="red", linestyle="--", label=f"Mean: {acqf_var_results.mean():.6f}")
axes[0, 1].axvline(
    np.median(acqf_var_results), color="green", linestyle="--", label=f"Median: {np.median(acqf_var_results):.6f}"
)
axes[0, 1].set_xlabel("Acquisition Function Results")
axes[0, 1].set_ylabel("Frequency")
axes[0, 1].set_title("Distribution of Acquisition Function Results")
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. Histogram of differences
axes[1, 0].hist(differences_var, bins=8, alpha=0.7, color="orange", edgecolor="black")
axes[1, 0].axvline(differences_var.mean(), color="red", linestyle="--", label=f"Mean: {differences_var.mean():.6f}")
axes[1, 0].axvline(0, color="black", linestyle="--", alpha=0.8, label="Perfect Agreement")
axes[1, 0].set_xlabel("Difference (QoI - AcqF)")
axes[1, 0].set_ylabel("Frequency")
axes[1, 0].set_title("Distribution of Differences")
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 4. Comparative box plot
box_data = [qoi_var_results, acqf_var_results]
box_labels = ["QoI Variance", "AcqF Variance"]
bp = axes[1, 1].boxplot(box_data, labels=box_labels, patch_artist=True)
bp["boxes"][0].set_facecolor("lightblue")
bp["boxes"][1].set_facecolor("lightgreen")
axes[1, 1].set_ylabel("Values")
axes[1, 1].set_title("Box Plot Comparison")
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# %%
test_exp.fetch_data().df.tail()  # type: ignore[reportAttributeAccessIssue]

# %%
original_test_exp.fetch_data().df.tail()  # type: ignore[reportAttributeAccessIssue]

# %%
# TODO:
# - Convert form model space to problem space
# - Add point to experiment
# - run QOI
# - score should be equal to acqf(torch.tensor([[[0.5, 0.5]]]))
# - Clean up and document code
# back to other list

# %%
#### SURFACE PLOT OF ACQUISITION FUNCTION


# %% Perform the grid search and plot
point_per_dim = 21
# Acquisition function operates in the model space, so we feed the model space to the acquisition function.
Hs = torch.linspace(0, 1, point_per_dim)
Tp = torch.linspace(0, 1, point_per_dim)
grid_hs, grid_tp = torch.meshgrid(Hs, Tp, indexing="xy")
# Convert from model space [0,1] to problem space to check constraints
# Get the search space parameters from your experiment
search_space_parameters = test_exp.search_space.parameters  # or use SEARCH_SPACE.parameters
Hs_param = search_space_parameters["Hs"]
Tp_param = search_space_parameters["Tp"]

# Transform to problem space
hs_problem = Hs_param.lower + grid_hs * (Hs_param.upper - Hs_param.lower)
tp_problem = Tp_param.lower + grid_tp * (Tp_param.upper - Tp_param.lower)

# Apply constraint: Hs <= 1.5*Tp - 1
constraint_mask = hs_problem <= (1.5 * tp_problem - 1.0)

# Create valid grid points
grid = torch.stack([grid_hs, grid_tp], dim=-1)
valid_indices = torch.where(constraint_mask)

# Get only the valid points for evaluation
valid_grid_points = grid[valid_indices]  # Shape: (n_valid_points, 2)
x_candidates = valid_grid_points.unsqueeze(1)  # Shape: (n_valid_points, 1, 2)

# Evaluate acquisition function on valid points only
scores_valid = acqf(x_candidates)

max_valid_score = scores_valid.max()

# Create full score grid with NaN for invalid points
scores = torch.full_like(grid_hs, torch.nan)
scores[valid_indices] = scores_valid.squeeze()

print(f"Total grid points: {point_per_dim**2}")
print(f"Valid points after constraint: {len(scores_valid)}")
print(f"Percentage valid: {len(scores_valid) / (point_per_dim**2) * 100:.1f}%")


# Find the point with the maximum score
max_score_idx = scores_valid.argmax()

# Get the corresponding point in model space
max_point_model_space = valid_grid_points[max_score_idx]

# Convert to problem space
max_point_problem_space = torch.tensor(
    [
        Hs_param.lower + max_point_model_space[0] * (Hs_param.upper - Hs_param.lower),
        Tp_param.lower + max_point_model_space[1] * (Tp_param.upper - Tp_param.lower),
    ]
)

# %%
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")
_ = ax.view_init(elev=30, azim=45)  # type: ignore[attr-defined]  # pyright: ignore[reportUnnecessaryTypeIgnore]
_ = ax.plot_surface(grid_hs, grid_tp, scores, cmap="viridis", edgecolor="none")  # type: ignore[attr-defined] # pyright: ignore[reportAttributeAccessIssue]
_ = ax.set_xlabel("Hs")  # type: ignore[assignment]
_ = ax.set_ylabel("Tp")  # type: ignore[assignment]
_ = ax.set_zlabel("score")  # type: ignore[attr-defined] # pyright: ignore[reportAttributeAccessIssue]
_ = ax.set_title("Score surface plot")  # type: ignore[assignment]
print("max_score ", max_valid_score.item())
print(f"Max point (model space): Hs={max_point_model_space[0]:.4f}, Tp={max_point_model_space[1]:.4f}")
print(f"Max point (problem space): Hs={max_point_problem_space[0]:.4f}, Tp={max_point_problem_space[1]:.4f}")

# %%
# Visualize current surface
fig_test_exp = plot_gp_fits_2d_surface_from_experiment(test_exp, 64, metrics=true_loc_scale_function_estimates)
fig_test_exp.show()

# %%
# Add the maximum point to the experiment
test_exp_copy = copy.deepcopy(test_exp)
trial = test_exp_copy.new_trial()
trial.add_arm(
    Arm(
        parameters={
            "Hs": max_point_problem_space[0].item(),
            "Tp": max_point_problem_space[1].item(),
        }
    )
)
_ = trial.run()
_ = trial.mark_completed()


# %%
## Run the acquisition function on the new model with the new point added
model = Models.BOTORCH_MODULAR(
    experiment=test_exp_copy,
    data=test_exp_copy.fetch_data(metrics=list(test_exp_copy.optimization_config.metrics.values())),  # type: ignore  # noqa: PGH003
    fit_tracking_metrics=False,  # Needed for QoIMetric to work properly
)
input_transform, outcome_transform = transforms.ax_to_botorch_transform_input_output(
    transforms=list(model_bridge_only_model.transforms.values()), outcome_names=model_bridge_only_model.outcomes
)


# %%
test_exp.fetch_data().df.tail()  # type: ignore[reportAttributeAccessIssue]

# %%
test_exp_copy.fetch_data().df.tail()  # type: ignore[reportAttributeAccessIssue]

# %%
acqf_new = ExperimentBasedAcquisition(model.model.surrogate.model, QOI_ESTIMATOR, test_exp_copy, input_transform)

# %%
scores_valid = acqf_new(x_candidates)

max_valid_score = scores_valid.max()

# Create full score grid with NaN for invalid points
scores = torch.full_like(grid_hs, torch.nan)
scores[valid_indices] = scores_valid.squeeze()

print(f"Total grid points: {point_per_dim**2}")
print(f"Valid points after constraint: {len(scores_valid)}")
print(f"Percentage valid: {len(scores_valid) / (point_per_dim**2) * 100:.1f}%")


# Find the point with the maximum score
max_score_idx = scores_valid.argmax()

# Get the corresponding point in model space
max_point_model_space = valid_grid_points[max_score_idx]

# Convert to problem space
max_point_problem_space = torch.tensor(
    [
        Hs_param.lower + max_point_model_space[0] * (Hs_param.upper - Hs_param.lower),
        Tp_param.lower + max_point_model_space[1] * (Tp_param.upper - Tp_param.lower),
    ]
)

# %%
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")
_ = ax.view_init(elev=30, azim=45)  # type: ignore[attr-defined]  # pyright: ignore[reportUnnecessaryTypeIgnore]
_ = ax.plot_surface(grid_hs, grid_tp, scores, cmap="viridis", edgecolor="none")  # type: ignore[attr-defined] # pyright: ignore[reportAttributeAccessIssue]
_ = ax.set_xlabel("Hs")  # type: ignore[assignment]
_ = ax.set_ylabel("Tp")  # type: ignore[assignment]
_ = ax.set_zlabel("score")  # type: ignore[attr-defined] # pyright: ignore[reportAttributeAccessIssue]
_ = ax.set_title("Score surface plot")  # type: ignore[assignment]
print("max_score ", max_valid_score.item())
print(f"Max point (model space): Hs={max_point_model_space[0]:.4f}, Tp={max_point_model_space[1]:.4f}")
print(f"Max point (problem space): Hs={max_point_problem_space[0]:.4f}, Tp={max_point_problem_space[1]:.4f}")


# %%
from importlib import reload

import optimal_doe
from optimal_doe import ExperimentBasedAcquisition  # noqa: E402

reload(optimal_doe)

exp_acqf_class = ExperimentBasedAcquisition


def exp_e_look_ahead_generator_run(experiment: Experiment) -> GeneratorRun:  # noqa: D103
    # Fist building model to get the transforms
    # TODO (se -2024-11-20): This refits hyperparameter each time, we don't want to do this.

    # TODO(@henrikstoklandberg 2025-04-29): Ticket "Transforms to work with QoI metric" adress this issue.
    # The problem is that transform.Ymean.keys is dict_keys(['loc', 'scale', 'QoIMetric'])
    # after the QoI metric is inculuded in the experiment. Then you get a error from line 249 in
    # transform.py. Was not able to figure out how to fix this in the time I had.
    # Ideally we should find a way to only use data=experiment.fetch_data() in the model_bridge_only_model.
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
        botorch_acqf_class=exp_acqf_class,
        fit_tracking_metrics=False,  # Needed for QoIMetric to work properly
        acquisition_options={
            # "model": model_bridge_only_model.model.surrogate.model,  # type: ignore[reportAttributeAccessIssue]
            "qoi_estimator": QOI_ESTIMATOR,
            "experiment": experiment,
            "input_transform": input_transform,
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


# %%
# run optimal look ahead experiment
exp_e_look_ahead = make_exp()

# Add the QoI metric to the experiment
_ = exp_e_look_ahead.add_tracking_metric(QOI_METRIC)

# This needs to be instantiated outside of the loop so the internal state of the generator persists.
sobol = Models.SOBOL(search_space=exp_e_look_ahead.search_space, seed=15)  # 5)

run_trials(
    experiment=exp_e_look_ahead,
    warm_up_generator=create_sobol_generator(sobol),
    doe_generator=exp_e_look_ahead_generator_run,
    warm_up_runs=warm_up_runs,
    doe_runs=4,
)


# %%

# %%
