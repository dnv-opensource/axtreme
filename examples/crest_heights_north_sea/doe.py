# %%  # noqa: D100
from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from ax import (
    Experiment,
    SearchSpace,
)
from ax.core import GeneratorRun, ParameterType, RangeParameter
from ax.modelbridge import ModelBridge
from ax.modelbridge.registry import Models
from botorch.optim import optimize_acqf
from brute_force_loc_and_scale_estimates import (  # type: ignore[import-not-found]
    get_brute_force_loc_and_scale_functions,
)
from numpy.typing import NDArray
from problem import (  # type: ignore[import-not-found]
    DIST,
    # make_exp,
    period_length,
    # sim,
)
from simulator import (  # type: ignore[import-not-found]
    MaxCrestHeightSimulatorSeeded,
    max_crest_height_simulator_function,
)
from torch.utils.data import DataLoader
from usecase.env_data import collect_data  # type: ignore[import-not-found]

from axtreme import sampling
from axtreme.acquisition import QoILookAhead
from axtreme.data import FixedRandomSampler, MinimalDataset
from axtreme.experiment import add_sobol_points_to_experiment, make_experiment
from axtreme.metrics import QoIMetric
from axtreme.plotting.doe import plot_qoi_estimates_from_experiment
from axtreme.plotting.gp_fit import plot_gp_fits_2d_surface_from_experiment
from axtreme.qoi import MarginalCDFExtrapolation
from axtreme.sampling.ut_sampler import UTSampler
from axtreme.utils import transforms

torch.set_default_dtype(torch.float64)
device = "cpu"

# pyright: reportUnnecessaryTypeIgnoreComment=false

# %%
# TODO(@henrikstoklandberg 2025-04-28): Update the search space to match the problem once we decide on the search space.
# For now a square search space is used, as we get Nan values from the simulator if we use the current search space in
# problem.py as of 2025-04-28.
search_space = SearchSpace(
    parameters=[
        RangeParameter(name="Hs", parameter_type=ParameterType.FLOAT, lower=7.5, upper=20),
        RangeParameter(name="Tp", parameter_type=ParameterType.FLOAT, lower=7.5, upper=20),
    ]
)

# Seeded simulator function for reproduceable results
sim = MaxCrestHeightSimulatorSeeded()


# To handle the difference in search space configuration between the problem and the DOE, a custom make experiment
# function is also needed.
def make_exp() -> Experiment:
    """Convenience function returns a fresh Experiement of this problem."""
    # n_simulations_per_point can be changed, but it is typically a good idea to set it here so all QOIs and Acqusition
    # Functions are working on the same problem and are comparable
    return make_experiment(sim, search_space, DIST, n_simulations_per_point=10)  # 1_000)


dist = DIST


# %%
raw_data: pd.DataFrame = collect_data()

# we convert this data to Numpy for ease of use from here on out
env_data: NDArray[np.float64] = raw_data.to_numpy()

# %%
# Bruteforce estimate placeholder
# TODO(@henrikstoklandberg 2025-05-09): Add more sophisticated brute force estimate of the QoI when ready
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
    """Helper function for running trials for an experiment and returning the QOI results using QoI metric.

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
n_iter = 15  # 40
warm_up_runs = 3

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
sobol = Models.SOBOL(search_space=exp_sobol.search_space, seed=5)


def create_sobol_generator(sobol: ModelBridge) -> Callable[[Experiment], GeneratorRun]:
    """Closure helper to run a sobol generator in the interface run_trails required.

    Note the typing is a bit general -> should be a sobol generatror.

    Returns:
        Callable[[Experiment], GeneratorRun]: A function that takes an experiment and returns a generator run.
    """

    def sobol_generator_run(_: Experiment) -> GeneratorRun:
        return sobol.gen(1)

    return sobol_generator_run


sobol_generator_run = create_sobol_generator(sobol)

run_trials(
    experiment=exp_sobol,
    warm_up_generator=sobol_generator_run,
    doe_generator=sobol_generator_run,
    warm_up_runs=warm_up_runs,
    doe_runs=n_iter,
)


# %% [markdown]
# ###  Load brute force loc and scale function estimates from saved data file
# %%
# Get brute force loc and scale functions from saved data
true_loc_scale_function_estimates = get_brute_force_loc_and_scale_functions(search_space)


# %%
# Plot the surface against brute force loc and scale function estimates for some trials
fig_trial_warm_up = plot_gp_fits_2d_surface_from_experiment(
    exp_sobol, warm_up_runs, metrics=true_loc_scale_function_estimates
)
fig_trial_warm_up.show()
fig_last_trial = plot_gp_fits_2d_surface_from_experiment(exp_sobol, n_iter, metrics=true_loc_scale_function_estimates)
fig_last_trial.show()

# %%
# Save the plot for documentation
fig_last_trial.write_html("results/doe/plots/sobol_gp_vs_true_functions.html")


# %% [markdown]
# ### Lookahead acquisition function
# The below is a test of the optimation of the acquisition function. It is not a full DOE, but rather a test to see what
# the acquisition function surface looks like for a single run and if the optimization settings are reasonable.
# TODO(@henrikstoklandberg 2025-04-28): After experimenting with the acquisition function, this could or should be moved
# to a different file.

# %%
# Define the model to use.
exp = make_exp()
add_sobol_points_to_experiment(exp, n_iter=64, seed=8)
# Use ax to create a gp from the experiment
botorch_model_bridge = Models.BOTORCH_MODULAR(
    experiment=exp,
    data=exp.fetch_data(),
)

# We need to collect the transforms used to the model gives result in the problem/outcome space.
input_transform, outcome_transform = transforms.ax_to_botorch_transform_input_output(
    transforms=list(botorch_model_bridge.transforms.values()), outcome_names=botorch_model_bridge.outcomes
)
QOI_ESTIMATOR.input_transform = input_transform
QOI_ESTIMATOR.outcome_transform = outcome_transform

model = botorch_model_bridge.model.surrogate.model


# %% How long does a single run take
acqusition = QoILookAhead(model, QOI_ESTIMATOR)
scores = acqusition(torch.tensor([[[0.5, 0.5]]]))

# %% Perform the grid search and plot
point_per_dim = 21

# Acquisition function operates in the model space, so we feed the model space to the acquisition function.
Hs = torch.linspace(0, 1, point_per_dim)
Tp = torch.linspace(0, 1, point_per_dim)
grid_hs, grid_tp = torch.meshgrid(Hs, Tp, indexing="xy")
grid = torch.stack([grid_hs, grid_tp], dim=-1)
# make turn into a shape that can be processsed by the acquisition function
x_candidates = grid.reshape(-1, 1, 2)
acqusition = QoILookAhead(model, QOI_ESTIMATOR)
scores = acqusition(x_candidates)
scores = scores.reshape(grid.shape[:-1])


# %%
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")
_ = ax.view_init(elev=30, azim=45)  # type: ignore[attr-defined]  # pyright: ignore[reportUnnecessaryTypeIgnore]
_ = ax.plot_surface(grid_hs, grid_tp, scores, cmap="viridis", edgecolor="none")  # type: ignore[attr-defined] # pyright: ignore[reportAttributeAccessIssue]
_ = ax.set_xlabel("Hs")  # type: ignore[assignment]
_ = ax.set_ylabel("Tp")  # type: ignore[assignment]
_ = ax.set_zlabel("score")  # type: ignore[attr-defined] # pyright: ignore[reportAttributeAccessIssue]
_ = ax.set_title("Score surface plot")  # type: ignore[assignment]
print("max_score ", scores.max())


# %% perform a round of optimisation using the under the hood optimiser
candidate, result = optimize_acqf(
    acqusition,
    bounds=torch.tensor([[0.0, 0.0], [1.0, 1.0]]),
    q=1,
    num_restarts=20,
    raw_samples=50,
    options={
        "with_grad": False,  # True by default.
        "method": "Nelder-Mead",  # "L-BFGS-B" by default
        "maxfev": 5,
    },
    retry_on_optimization_warning=False,
)
print(candidate, result)

# %% [markdown]
# #### Lookahead acquisition function
# Define the helpers to run the custom acquisition function.

# %%
acqf_class = QoILookAhead


def look_ahead_generator_run(experiment: Experiment) -> GeneratorRun:  # noqa: D103
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
        botorch_acqf_class=acqf_class,
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
# Run the DOE

# %%
exp_look_ahead = make_exp()

# Add the QoI metric to the experiment
_ = exp_look_ahead.add_tracking_metric(QOI_METRIC)

# This needs to be instantiated outside of the loop so the internal state of the generator persists.
sobol = Models.SOBOL(search_space=exp_look_ahead.search_space, seed=5)

run_trials(
    experiment=exp_look_ahead,
    warm_up_generator=create_sobol_generator(sobol),
    doe_generator=look_ahead_generator_run,
    warm_up_runs=warm_up_runs,
    doe_runs=n_iter,
)

# %%
# Plot the surface against brute force loc and scale function estimates for some trials
fig_trial_warm_up = plot_gp_fits_2d_surface_from_experiment(
    exp_look_ahead, warm_up_runs, metrics=true_loc_scale_function_estimates
)
fig_trial_warm_up.show()
fig_last_trial = plot_gp_fits_2d_surface_from_experiment(
    exp_look_ahead, n_iter, metrics=true_loc_scale_function_estimates
)
fig_last_trial.show()

# %%
# Save the plot for documentation
fig_last_trial.write_html("results/doe/plots/doe_gp_vs_true_functions.html")


# %% [markdown]
# ### Plot the results


# %%
ax = plot_qoi_estimates_from_experiment(exp_sobol, name="Sobol")
ax = plot_qoi_estimates_from_experiment(exp_look_ahead, ax=ax, color="green", name="look ahead")
_ = ax.axhline(brute_force_qoi_estimate, c="black", label="brute_force_value")  # type: ignore[assignment]
_ = ax.set_xlabel("Number of DOE iterations")  # type: ignore[assignment]
_ = ax.set_ylabel("Response")  # type: ignore[assignment]
_ = ax.legend()  # type: ignore[assignment]
