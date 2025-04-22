# noqa: ALL
# type: ignore
# ruff: noqa: ERA001 PGH003 PGH004 TD002 F841
# %% [markdown]
# # Basic example
# This notebook demonstrates using `axtreme` to solve a toy problem.
#
#
# The following creates a toy problem, and calculates the brute force solution. We then demonstrate how `axtreme` can be
#  used to achieve the same results while running the simulator far fewer times. Specially, we show how to:
# - Define the problem in the Ax framework.
# - Create a surrogate model (automatically using ax).
# - Estimate the QoI.
# - Use DoE to reduce uncertainty.
#
#
# > NOTE: This is an introductory example intended to provide an overview of the process. As such, a number of
# simplification are made. More in depth tutorial are provided here at a later date.

# %%
import sys
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch
from ax import (
    Experiment,
    SearchSpace,
)
from ax.core import GeneratorRun, ObservationFeatures, ParameterType, RangeParameter
from ax.modelbridge.registry import Models
from botorch.optim import optimize_acqf
from matplotlib.axes import Axes
from numpy.typing import NDArray
from plotly.subplots import make_subplots
from scipy.stats import gumbel_r
from torch.distributions import Normal
from torch.utils.data import DataLoader

from axtreme import sampling
from axtreme.acquisition import QoILookAhead
from axtreme.data import FixedRandomSampler, MinimalDataset
from axtreme.experiment import add_sobol_points_to_experiment, make_experiment
from axtreme.plotting.gp_fit import plot_gp_fits_2d_surface, plot_surface_over_2d_search_space
from axtreme.plotting.histogram3d import histogram_surface3d
from axtreme.qoi import MarginalCDFExtrapolation
from axtreme.qoi.qoi_estimator import QoIEstimator
from axtreme.sampling.ut_sampler import UTSampler
from axtreme.simulator import utils as sim_utils
from axtreme.utils import population_estimators, transforms

torch.set_default_dtype(torch.float64)
device = "cpu"

# pyright: reportUnnecessaryTypeIgnoreComment=false

# %% [markdown]
# ## Explore the toy problem
# The `axtreme` package expects 2 key pieces of input:
# - A simulator:
# - Samples from the environment
#
# Here we use a mock simulator and env data defined in `examples`. The following section explores these raw inputs.

# %%
root_dir = Path("../")
sys.path.append(str(root_dir))
from examples.demo2d.problem.brute_force import collect_or_calculate_results
from examples.demo2d.problem.env_data import collect_data
from examples.demo2d.problem.simulator import (
    DummySimulatorSeeded,
    _true_loc_func,
    _true_scale_func,
    dummy_simulator_function,
)

# %% [markdown]
# ### Simulator
# The toy simulator in this example uses a gumbel distribution as its noise model. A gumbel distribution has 2
# parameters, location (loc) and scale. The toy problem has one underlying function that controls the loc,
# and another to control the scale. This can be written as:
# - $y = loc(x) + noise, where noise ~ Gumbel(0, scale(x))$
# - OR
# - $y = Gumbel(loc(x), scale(x)).sample()$
#
# In a real problem the underlying function that controls the output distribution would be unknown, but in this toy
# example we plot them directly to give a better understanding of the problem being solved in this example.

# %%
# plot it
fig = make_subplots(
    rows=1,
    cols=3,
    specs=[[{"type": "surface"}, {"type": "surface"}, {"type": "surface"}]],
    subplot_titles=("location", "scale", "Gumbel response surface (at q = [.1, .5, .9])"),
)


plot_search_space = SearchSpace(
    parameters=[
        RangeParameter(name="x1", parameter_type=ParameterType.FLOAT, lower=0, upper=1),
        RangeParameter(name="x2", parameter_type=ParameterType.FLOAT, lower=0, upper=1),
    ]
)

# plot the underling location and scale function
_ = fig.add_trace(plot_surface_over_2d_search_space(plot_search_space, funcs=[_true_loc_func]).data[0], row=1, col=1)
_ = fig.add_trace(plot_surface_over_2d_search_space(plot_search_space, funcs=[_true_scale_func]).data[0], row=1, col=2)


# Plot the response surface at different quantiles
def gumbel_helper(x: np.ndarray[tuple[int, int], np.dtype[np.float64]], q: float = 0.5) -> NDArray[np.float64]:
    return gumbel_r.ppf(q=q, loc=_true_loc_func(x), scale=_true_scale_func(x))


quantile_plotters: list[Callable[[np.ndarray[Any, np.dtype[np.float64]]], np.ndarray[Any, np.dtype[np.float64]]]] = [
    partial(gumbel_helper, q=q) for q in [0.1, 0.5, 0.9]
]
response_distribution = plot_surface_over_2d_search_space(plot_search_space, funcs=quantile_plotters)
_ = [fig.add_trace(data, row=1, col=3) for data in response_distribution.data]

# label the plot
_ = fig.update_scenes({"xaxis": {"title": "x1"}, "yaxis": {"title": "x2"}, "zaxis": {"title": "response"}})
_ = fig.update_traces(showscale=False)
fig.show()

# %% [markdown]
# ### Environment
# The other input required by the `axtreme` package is static dataset of environment samples. Here we load and plot the
# data we will be using in this example.

# %%
import pandas as pd

raw_data: pd.DataFrame = collect_data()

# we convert this data to Numpy for ease of use from here on out
env_data: NDArray[np.float64] = raw_data.to_numpy()
fig = histogram_surface3d(env_data)
_ = fig.update_layout(title_text="Environment distribution estimate from samples")
_ = fig.update_layout(scene_aspectmode="cube")
fig.show()

# %% [markdown]
# ### Define the problem
# Because we are using a toy example we can directly calculate the ERD and our QOI. This is the answer that we are
# trying to recover using the `axtreme` package (in which we make minimal use of the simulator).
#
# We define the time span over which we wish to calculate the Extreme Response, and produce a brute force estimate.

# %%
# define the time span
N_ENV_SAMPLES_PER_PERIOD = 1000


n_erd_samples = 1000
erd_samples = []
for _ in range(n_erd_samples):
    indices = np.random.choice(env_data.shape[0], size=N_ENV_SAMPLES_PER_PERIOD, replace=True)  # noqa: NPY002
    period_sample = env_data[indices]

    responses = dummy_simulator_function(period_sample)
    erd_samples.append(responses.max())

_, axes = plt.subplots(ncols=2, figsize=(12, 5))

# plot the ERD distribution
_ = axes[0].hist(erd_samples, bins=100, density=True)
population_median_est_dist = population_estimators.sample_median_se(torch.tensor(erd_samples))
_ = ax_twin = axes[0].twinx()
_ = population_estimators.plot_dist(population_median_est_dist, ax=ax_twin, c="orange", label="QOI estimate")
_ = ax_twin.set_ylabel("Estimated PDF")
_ = ax_twin.legend()
_ = axes[0].set_title(
    f"Extreme response distribution\n"
    f"(each result represents the largest \nresponse seen {N_ENV_SAMPLES_PER_PERIOD} length period)"
)
_ = axes[0].set_xlabel("Response value")
_ = axes[0].set_ylabel("Density")
_ = axes[0].grid(visible=True)

# Plot the estimated QoI distribution
_ = population_estimators.plot_dist(population_median_est_dist, ax=axes[1], c="orange")
_ = axes[1].set_title("Sample estimate \nof population median (QOI)")
_ = axes[1].set_xlabel("Response value")
_ = axes[1].grid(visible=True)


brute_force_qoi_estimate = np.median(erd_samples)
print(f"Brute force estimate of our QOI is {brute_force_qoi_estimate}")

# %%
# For this specific problem we have precalculated a large number of brute force ERD samples.
# This allows up to treat the brute_force_qoi_estimate as a point for the purpose of this tutorial.
erd_samples = collect_or_calculate_results(N_ENV_SAMPLES_PER_PERIOD, 300_000)
brute_force_qoi_estimate = np.median(erd_samples)
print(f"Brute force estimate of our QOI is {brute_force_qoi_estimate}")

# %% [markdown]
# # Using `axtreme` to solve the problem
# In the section above we have explored the key inputs to the `axtreme` package (Simulator and Environment data), and
# calculated the brute force answer to our QoI. In this section we will show how `axtreme` can be used to achieve the
# same result, while running the simulator far fewer times.
#
# `axtreme` is comprised of 4 main step:
# - Define the problem in the Ax framework:
# - Create a surrogate model: (this happens automatically once the problem has been defined in Ax).
# - Estimate the Qoi: Estimate the QoI (and our confidence) using the surrogate model.
# - DoE: reduce uncertainty in the QoI while running the simulator as little as possible.
#

# %% [markdown]
# ## Define the problem in the `ax` framework.
# The following steps need to be taken to define the problem in the Ax framework:
# - Ensure the simulator conforms to the required interface.
# - Decide on the search space to use.
# - Pick a distribution that you believe captures the noise behaviour of your simulator.
#
# These decisions are straight forward for this toy example. Advice on how to choose these parameters in more real world
# problems are provided in other tutorials (TODO(sw 2024-11-21): include these tutorials).

# %% [markdown]
# ### Make our simulator conform to the required interface

# %%
# Check if it complies
print(sim_utils.is_valid_simulator(dummy_simulator_function, verbose=True))
# use a helper to add the 'n_simulations_per_point' functionality
sim = sim_utils.simulator_from_func(dummy_simulator_function)
# check that it now complies
print(sim_utils.is_valid_simulator(sim, verbose=True))

# NOTE: for the rest of this notebook we proceed with a seeded simulator. This is exactly the same as
# dummy_simulator_function except it makes our results reproducible.
sim = DummySimulatorSeeded()
print(sim_utils.is_valid_simulator(sim, verbose=True))

# %% [markdown]
# ### Pick the search space over which to create a surrogate

# %%
search_space = SearchSpace(
    parameters=[
        RangeParameter(name="x1", parameter_type=ParameterType.FLOAT, lower=0, upper=1),
        RangeParameter(name="x2", parameter_type=ParameterType.FLOAT, lower=0, upper=1),
    ]
)

# %% [markdown]
# ### Pick a distibution that you belive captures the noise behaviour of your simulator

# %%
dist = gumbel_r

# %% [markdown]
# ## Automatically set up you experiment
# Use the sim, search_space, and dist defined above to set up the `ax` `Experiment`.


# %%
def make_exp() -> Experiment:
    """Helper to ensure we always create an experiment with the same settings (so results are comparable)."""
    return make_experiment(sim, search_space, dist, n_simulations_per_point=200)


exp = make_exp()

# %% [markdown]
# ## Create a Surrogate model:
#
# Once we have defined an Ax experiment, we can use `ax` to create a surrogate. This surrogate provides the location and
# scale parameters for our gumbel distribution.
#
# We first need to add some training data to the experiment. We do this by passing the experiment some initial x points.
# Behind the scenes, `ax` will run the Simulator to obtain estimates for the location and scale parameters at those
# points. It will then use that data to create a surrogate model.

# %%
# Add random x points to the experiment
add_sobol_points_to_experiment(exp, n_iter=30, seed=8)

# Use ax to generate a surrogate
botorch_model_bridge = Models.BOTORCH_MODULAR(
    experiment=exp,
    data=exp.fetch_data(),
)

pred_mean, pred_covariance = botorch_model_bridge.predict([ObservationFeatures(parameters={"x1": 0.5, "x2": 0.5})])

# Lets compare the output distribution of the surrogate and the simulator at this point
pred_dist = dist(loc=pred_mean["loc"], scale=pred_mean["scale"])
simulator_samples = sim(np.array([[0.5, 0.5]]), n_simulations_per_point=200).flatten()


x_points = np.linspace(simulator_samples.min(), simulator_samples.max(), 100)

_ = plt.hist(simulator_samples, bins=len(simulator_samples) // 9, density=True, label="Simulator")
_ = plt.plot(x_points, pred_dist.pdf(x_points), label="Surrogate")  # pyright: ignore[reportAttributeAccessIssue]
_ = plt.xlabel("Response value")
_ = plt.ylabel("pdf")
_ = plt.title("Surrogate distribution vs Simulator distribution at point x = [0.5, 0.5]")
_ = plt.legend()
plt.show()

# %% [markdown]
# The surrogate also contains uncertainty about its estimate. Lets plot the other distributions that the surrogate
# believes are possible.

# %%
mean = np.array([pred_mean["loc"], pred_mean["scale"]])
mean = mean.flatten()

covariance = np.array(
    [
        [pred_covariance["loc"]["loc"], pred_covariance["loc"]["scale"]],
        [pred_covariance["scale"]["loc"], pred_covariance["scale"]["scale"]],
    ]
)
covariance = covariance.reshape(2, 2)

surrogate_distribution = scipy.stats.multivariate_normal(mean, covariance)
surrogate_distribution_samples = surrogate_distribution.rvs(size=5, random_state=6)
for sample in surrogate_distribution_samples:
    sample_dist = dist(loc=sample[0], scale=sample[1])
    _ = plt.plot(x_points, sample_dist.pdf(x_points), c="grey", alpha=0.5)  # pyright: ignore[reportAttributeAccessIssue]

_ = plt.plot(x_points, pred_dist.pdf(x_points), c="orange", label="Surrogate Mean")  # pyright: ignore[reportAttributeAccessIssue]
_ = plt.plot([], [], label="Posterior Samples", c="grey")  # hacky way to add a label
_ = plt.title("Distribution of possible responses at x = [0.5, 0.5]")
_ = plt.xlabel("Response value")
_ = plt.ylabel("pdf")
_ = plt.legend()
plt.show()

# %% [markdown]
# ## Estimate the QoI:
# Now that we have a surrogate model, we can use it to estimate the QoI. The uncertainty in our surrogate model should
# be reflected in our QoI estimate. Lets demonstrate how the estimate changes as we add more data to the surrogate model
# and it becomes more certain.
#
# In the following we make use of an existing QoI Estimator. `axtreme` provides a number of QoIEstimators for common
# tasks, but users can also create custom QoIEstimator for their specific problems. Details can be found
# (TODO(sw 2024-11-22): link to tutorial on how to create a custom QoIEstimator).
#
# In the following we demonstrate how the QoI estimate becomes more certain as the surrogate gets more training data.
#
# > NOTE: Training a Gp with `Models.BOTORCH_MODULAR` has inherit randomness that can't be turned off(e.g with
# `torch.manual_seed`). As a result there is slight randomness in the result even though all other seeds are set.

# %%
# A note on DataLoaders:
# We make use of Dataloader to manage providing data to the QoI. Env samples are only used in the QoI, so your env data
# can be in whatever format is supported by the QoI.
n_env_samples = 1_000
dataset = MinimalDataset(env_data)
sampler = FixedRandomSampler(dataset, num_samples=n_env_samples, seed=10, replacement=True)
dataloader = DataLoader(dataset, sampler=sampler, batch_size=256)

posterior_sampler = UTSampler()
# posterior_sampler = NormalIndependentSampler(torch.Size([n_posterior_samples])

qoi_estimator = MarginalCDFExtrapolation(
    # random dataloader give different env samples for each instance
    env_iterable=dataloader,
    period_len=N_ENV_SAMPLES_PER_PERIOD,
    quantile=torch.tensor(0.5),
    quantile_accuracy=torch.tensor(0.01),
    # IndexSampler needs to be used with GenericDeterministicModel. Each sample just selects the mean.
    posterior_sampler=posterior_sampler,
)

# %%
# Initialise and environment, get gp from `ax`, and calculate the QOI

n_training_points = [30, 50, 128, 512]
results = []

for points in n_training_points:
    exp = make_exp()
    add_sobol_points_to_experiment(exp, n_iter=points, seed=8)
    # Use ax to create a gp from the experiment
    botorch_model_bridge = Models.BOTORCH_MODULAR(
        experiment=exp,
        data=exp.fetch_data(),
    )

    # We need to collect the transforms used to the model gives result in the problem/outcome space.
    input_transform, outcome_transform = transforms.ax_to_botorch_transform_input_output(
        transforms=list(botorch_model_bridge.transforms.values()), outcome_names=botorch_model_bridge.outcomes
    )
    qoi_estimator.input_transform = input_transform
    qoi_estimator.outcome_transform = outcome_transform

    model = botorch_model_bridge.model.surrogate.model
    # reseed the dataloader each time so the dame dataset is used.
    results.append(qoi_estimator(model))


# %%
def get_mean_var(estimator: QoIEstimator, estimates: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """TODO: clean this up or delete.

    Args:
        estimator: the QoI function that produced the estimate
        estimates: (*b, n_estimator)

    Returns:
        tensor1: the mean of the estimates, with shape *b
        tensor1: the mean of the estimates, with shape *b

    """
    if not isinstance(estimates, torch.Tensor):  # pyright: ignore[reportUnnecessaryIsInstance]
        estimates = torch.tensor(estimates)

    mean = estimator.posterior_sampler.mean(estimates, -1)  # pyright: ignore[reportAttributeAccessIssue]
    var = estimator.posterior_sampler.var(estimates, -1)  # pyright: ignore[reportAttributeAccessIssue]

    return mean, var


# %%
_, axes = plt.subplots(nrows=len(n_training_points), sharex=True, figsize=(6, 6 * len(n_training_points)))

for ax, estimate, n_points in zip(axes, results, n_training_points, strict=True):
    mean, var = get_mean_var(qoi_estimator, torch.tensor(estimate))
    qoi_dist = Normal(mean, var**0.5)
    _ = population_estimators.plot_dist(qoi_dist, ax=ax, c="tab:blue", label="QOI estimate")

    ax.axvline(brute_force_qoi_estimate, c="orange", label="Brute force results")

    ax.set_title(f"QoI estimate with {n_points} training points")
    ax.set_xlabel("Response")
    ax.set_ylabel("Density")
    ax.legend()

# Note: if you are interested, the following allows you to see the result of the QoIEstimator if uncertainty is ignored.
# this may not be a very good estimate if the simulator mean doesn't match the surrogate.
# from axtreme.eval.qoi import qoi_ignoring_gp_uncertainty
# result_no_gp_uncertainty = qoi_ignoring_gp_uncertainty(qoi_estimator_gp_brute_force, model)

# %% [markdown]
# ## DoE: efficiently reduce uncertainty in the QoI.
#
# In the plots above we see QoI uncertainty can be reduced by adding training data. In this section we explore if we can
# obtain a similar reduction in uncertainty with fewer training point. We do this by picking new points intelligently.
#
# First we perform DoE using a space filling design (Sobol). This is the baseline that DoE should be able to improve on.
# We then run DoE using our custom acquisition function (QoILookAhead), and compare results.

# %% [markdown]
# ### Helper for runnig experiments


# %%
def run_trials(
    experiment: Experiment,
    warm_up_generator: Callable[[Experiment], GeneratorRun],
    doe_generator: Callable[[Experiment], GeneratorRun],
    qoi_estimator: MarginalCDFExtrapolation,
    warm_up_runs: int = 3,
    doe_runs: int = 15,
    qoi_iter: int = 1,
) -> NDArray[np.float64]:
    """Helper function for running many trials for an experiment and returning the QOI results.

    Args:
        experiment: Experiment to perform DOE on.
        warm_up_generator: Generator to create the initial training data on the experiment (e.g., Sobol).
        doe_generator: The generator being used to perform the DoE.
        qoi_estimator: The function to estimate the QOI after new data points are added to the experiment.
        warm_up_runs: Number of warm-up runs to perform before starting the DoE.
        doe_runs: Number of DoE runs to perform.
        qoi_iter: How often to calculate the QOI. If set to 1, the QOI will be calculated after every run.

    Returns:
        np.ndarray: Array of shape (n_qoi_iter, qoi_estimator_output) where:
            - n_qoi_iter: The number of times the qoi_estimator was called (on a new amount of data).
            - qoi_estimator_output: The results given on that dataset.
    """
    figs = []
    qoi_results = []
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

        model_bridge = Models.BOTORCH_MODULAR(
            experiment=experiment,
            data=experiment.fetch_data(),
        )
        if i % qoi_iter == 0:
            input_transform, outcome_transform = transforms.ax_to_botorch_transform_input_output(
                transforms=list(model_bridge.transforms.values()), outcome_names=model_bridge.outcomes
            )

            qoi_estimator.input_transform = input_transform
            qoi_estimator.outcome_transform = outcome_transform

            qoi_samples = qoi_estimator(model=model_bridge.model.surrogate.model)
            qoi_results.append(qoi_samples.detach().numpy())
        if i in (0, doe_runs):
            figs.append(
                plot_gp_fits_2d_surface(model_bridge, search_space, {"loc": _true_loc_func, "scale": _true_scale_func})
            )
        print(f"iter {i} done")

    for fig in figs:
        fig.show()
    return np.vstack(qoi_results)


# %% [markdown]
# How many iterations to run in the following DOEs

# %%
n_iter = 40
qoi_iter = 1
warm_up_runs = 3

# %% [markdown]
# ### Sobol model
# Surrogate trained without and a acquisition function as a comparative baseline.

# %%
exp_sobol = make_exp()
# This needs to be instantiated outside of the loop so the internal state of the generator persists.
sobol = Models.SOBOL(search_space=exp_sobol.search_space, seed=5)


def sobol_generator_run(_: Experiment) -> GeneratorRun:  # pyright: ignore[reportRedeclaration]
    return sobol.gen(1)


qoi_results_sobol = run_trials(
    experiment=exp_sobol,
    warm_up_generator=sobol_generator_run,
    doe_generator=sobol_generator_run,
    qoi_estimator=qoi_estimator,
    warm_up_runs=warm_up_runs,
    doe_runs=n_iter,
    qoi_iter=qoi_iter,
)

# %% [markdown]
# ### Custom acquisition function:
# We now set up the components required for the Custom acquisition function.
#
# ~~~
# NOTE: It is important to understand the behaviour of the acquisition function, as this has implication for the
# optimisation arguments used. The parameters set here are a robust option that do not assume a smooth function.
# If you can guarantee the acquisition function is smooth this should be updated to use more efficient methods.
# ~~~

# %% [markdown]
# #### Optional: Explore optimisation settings
#
# Prior to doing multiple rounds of optimisation, it can be useful to check the optimisation setting are appropriate
# for the acquisition function. Here we perform and plot a grid search, and compare it to the optimisation value to get
# a sense of the surface. This is more challenging to do in real problems.

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
qoi_estimator.input_transform = input_transform
qoi_estimator.outcome_transform = outcome_transform

model = botorch_model_bridge.model.surrogate.model

# %% How long does a single run take
acqusition = QoILookAhead(model, qoi_estimator)
scores = acqusition(torch.tensor([[[0.5, 0.5]]]))

# %% Perform the grid search and plot
point_per_dim = 21
x1 = torch.linspace(0, 1, point_per_dim)
x2 = torch.linspace(0, 1, point_per_dim)
grid_x1, grid_x2 = torch.meshgrid(x1, x2, indexing="xy")
grid = torch.stack([grid_x1, grid_x2], dim=-1)
# make turn into a shape that can be processsed by the acquisition function
x_candidates = grid.reshape(-1, 1, 2)
acqusition = QoILookAhead(model, qoi_estimator)
scores = acqusition(x_candidates)
scores = scores.reshape(grid.shape[:-1])

# %%
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")
_ = ax.view_init(elev=30, azim=45)  # pyright: ignore[reportAttributeAccessIssue]
_ = ax.plot_surface(grid_x1, grid_x2, scores, cmap="viridis", edgecolor="none")  # pyright: ignore[reportAttributeAccessIssue]
_ = ax.set_xlabel("x1")
_ = ax.set_ylabel("x2")
_ = ax.set_zlabel("score")  # pyright: ignore[reportAttributeAccessIssue]
_ = ax.set_title("Score surface plot")
print("max_score ", scores.max())

# %%
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
posterior_sampler = sampling.MeanSampler()


def look_ahead_generator_run(experiment: Experiment) -> GeneratorRun:
    # Fist building model to get the transforms
    # TODO (se -2024-11-20): This refits hyperparameter each time, we don't want to do this.
    model_bridge_only_model = Models.BOTORCH_MODULAR(
        experiment=experiment,
        data=experiment.fetch_data(),
    )
    input_transform, outcome_transform = transforms.ax_to_botorch_transform_input_output(
        transforms=list(model_bridge_only_model.transforms.values()), outcome_names=model_bridge_only_model.outcomes
    )
    # Adding the transforms to the QoI estimator
    qoi_estimator.input_transform = input_transform
    qoi_estimator.outcome_transform = outcome_transform

    # Building the model with the QoILookAhead acquisition function
    model_bridge_cust_ac = Models.BOTORCH_MODULAR(
        experiment=experiment,
        data=experiment.fetch_data(),
        botorch_acqf_class=acqf_class,
        acquisition_options={
            "qoi_estimator": qoi_estimator,
            "sampler": posterior_sampler,
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
# This needs to be instantiated outside of the loop so the internal state of the generator persists.
sobol = Models.SOBOL(search_space=exp_look_ahead.search_space, seed=5)


def sobol_generator_run(_: Experiment) -> GeneratorRun:  # pyright: ignore[reportRedeclaration]
    return sobol.gen(1)


qoi_results_look_ahead = run_trials(
    experiment=exp_look_ahead,
    warm_up_generator=sobol_generator_run,
    doe_generator=look_ahead_generator_run,
    qoi_estimator=qoi_estimator,
    warm_up_runs=warm_up_runs,
    doe_runs=n_iter,
    qoi_iter=qoi_iter,
)

# %% [markdown]
# ### Plot the results


# %%
def plot_raw_ut_estimates(
    mean: torch.Tensor,
    var: torch.Tensor,
    ax: None | Axes = None,
    points_between_ests: int = 1,
    name: str | None = None,
    **kwargs: Any,  # noqa: ANN401
) -> Axes:
    """NOTE very quick and dirty, assumes you know how to interpret the raw UT results (e.g rather than being given).

    Args:
        mean: shape (n,) mean qoi estimates for each run.
        var: shape (n,) variance qoi estimates for each run.
        ax: ax to add the plots to. If not provided, one will be created.
        points_between_ests: This should be used if multiple DoE iterations are used between qoi estimates
            (e.g if the estimate is expensive). It adjusts the scale of the x axis.
        name: optional name that should be added to the legend information for this plot
        kwargs: kwargs that should be passed to matplotlib. Must be applicable to `ax.plot` and `ax.fill_between`

    Returns:
        Axes: the ax with the plot.
    """
    if ax is None:
        _, ax = plt.subplots()

    x = range(1, (len(mean) + 1) * points_between_ests, points_between_ests)
    _ = ax.fill_between(
        x,
        mean - 1.96 * var**0.5,
        mean + 1.96 * var**0.5,
        label=f"90% Confidence Bound {name}",
        alpha=0.3,
        **kwargs,
    )

    _ = ax.plot(x, mean, **kwargs)

    return ax


# %%
mean, var = get_mean_var(qoi_estimator, qoi_results_sobol)
ax = plot_raw_ut_estimates(mean, var, name="Sobol")
mean, var = get_mean_var(qoi_estimator, qoi_results_look_ahead)
ax = plot_raw_ut_estimates(mean, var, ax=ax, color="green", name="look ahead")
_ = ax.axhline(brute_force_qoi_estimate, c="black", label="brute_force_value")
_ = ax.set_xlabel("Number of DOE iterations")
_ = ax.set_ylabel("Response")
_ = ax.legend()


# %% Make a grid
point_per_dim = 21
x1 = torch.linspace(0, 1, point_per_dim)
x2 = torch.linspace(0, 1, point_per_dim)
grid_x1, grid_x2 = torch.meshgrid(x1, x2, indexing="xy")
grid = torch.stack([grid_x1, grid_x2], dim=-1)
print("grid shape", grid.shape)
# print("grid", grid)
grid_np = grid.numpy()
print("grid_np shape", grid_np.shape)
print("grid_np[0] shape", grid_np[0].shape)
# turn into a shape that can be processsed by the acquisition function
x_candidates = grid.reshape(-1, 1, 2)
print("x_candidates shape", x_candidates.shape)
# get results from simulator
# Reshape the grid to match the simulator's expected input shape
grid_np_flat = grid_np.reshape(-1, 2)
print("grid_np_flat shape", grid_np_flat.shape)

# Run the simulator for each point in the grid
simulator_samples = sim(grid_np_flat, n_simulations_per_point=200)


print("simulator_samples shape", simulator_samples.shape)
print("simulator_samples min max", simulator_samples.min(), simulator_samples.max())
print("simulator_samples", simulator_samples[:10])


# %%
# Calculate the QoI (e.g., maximum response) for each grid point
qoi_values = simulator_samples.max(axis=1)

# Reshape the QoI values back to the grid shape for visualization
qoi_grid = qoi_values.reshape(grid_np.shape[:-1])
print("qoi_grid max", qoi_grid.max())

# Plot the QoI values as a surface
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")
_ = ax.view_init(elev=30, azim=45)
_ = ax.plot_surface(grid_x1, grid_x2, qoi_grid, cmap="viridis", edgecolor="none")
_ = ax.set_xlabel("x1")
_ = ax.set_ylabel("x2")
_ = ax.set_zlabel("QoI")
_ = ax.set_title("QoI Surface Plot")
plt.show()

# %%
"""OptimalLookAhead acquisition function that looks ahead at possible models and optimize according to a quantity of
interest."""

import warnings
from contextlib import ExitStack
from typing import Any

import torch
from ax.models.torch.botorch_modular.optimizer_argparse import _argparse_base, optimizer_argparse
from botorch.acquisition import AcquisitionFunction
from botorch.acquisition.input_constructors import acqf_input_constructor
from botorch.models import SingleTaskGP
from botorch.models.model import Model
from botorch.utils.transforms import t_batch_mode_transform
from gpytorch.likelihoods.gaussian_likelihood import FixedNoiseGaussianLikelihood, GaussianLikelihood

from axtreme.evaluation import EvaluationFunction
from axtreme.qoi.qoi_estimator import QoIEstimator
from axtreme.sampling import MeanSampler, PosteriorSampler
from axtreme.simulator.base import Simulator
from axtreme.utils.model_helpers import get_target_noise_singletaskgp, get_training_data_singletaskgp


class OptimalLookAhead(AcquisitionFunction):
    """QoILookAhead is a generic acquisition function that estimates the usefulness of a design point.

    It estimates the usefulness of a design point by:

    - Creating a new model(s) (e.g the looking ahead) by including the design point in the training data.

      .. Note::
        The new model condition on the design point, it does not update hyperparameters.

    - Calculate the QoI with the new model, and find the variance of this distribution.

    The design point that results in the lowest variance QoI is considered the most desireable.

    Note on Optimisation:
        Optimising the AcquisitionFunction is an important part of the DoE process. The stochasticity and smoothness of
        the acquisition function determine what optimisers can be used. This acquisition function has the following
        properties with the default setup:

    Smoothness:
        QoILookAhead is smooth (twice differentiable) if:

        - The model used for instantiation produces smooth outputs.
        - A sampler produces smooth y values.
        - The method used to produce y_var estimates is smooth.
        - The QoIEstimator is smooth with respect to the model (e.g small changes in the model produce smooth
          change in the QoIEstimator result.)

        The default optimiser assume the QoI may not be smooth, and uses a gradient free optimiser. If your QoI is
        smooth these setting should be overridden.

    Stochasticity:
        QoILookAhead is deterministic if all the above components are deterministic.
        With default settings it is deterministic.
    """

    def __init__(
        self,
        model: SingleTaskGP,
        qoi_estimator: QoIEstimator,
        simulator: Simulator,
        sampler: PosteriorSampler | None = None,
    ) -> None:
        """QoILookAhead acquisition function.

        Args:
            model: A fitted model. Only SingleTaskGP models are currently supported.

                - General GpytorchModel may eventually be supported.

            qoi_estimator: A callable conforming to QoIEstimator protocol.
            sampler: A sampler that is used to sample fantasy observations for each candidate point.
                If None, a MeanSampler is used. This then uses the mean of the posterior as the fantasy observation.

                .. Note::
                    Sampler choice can effect the stochasticty and smoothness of the acquisition function. See class
                    docs for details.
        """
        super().__init__(model)
        self.qoi = qoi_estimator

        if sampler is None:
            sampler = MeanSampler()
        self.sampler = sampler

        self.simulator = simulator

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: torch.Tensor) -> torch.Tensor:  # noqa: N803  # pyright: ignore[reportIncompatibleMethodOverride] (necessary due to `t_batch_mode_transform` decorator)
        """Forward method for the QoI acquisition function.

        For each candidate point in x this acquisition function does the following:

        - Fantasizes (via the chosen sampler) possible new observations (y) at this candidate point.
        - Trains a new GP with the additional (x,y_i) pair for each fantasy observation y_i.
        - Evaluates the qoi_estimator on these new GPs, and reports the variance in the resulting estimates.

        This optimization will pick the point that resulted in the lowest mean variance in the QoI estimates.

        Args:
            X: (t_batch, 1, d) input points to evaluate the acquisiton function at.

        Returns:
            The output of the QoI acquisition function that will be optimized with shape (num_points,).

        Todo:
            - This should be updated to use the `FantasizeMixin`. This is the botorch interface indicating
              `model.fantasize(X)` is supported, which does a large chunk of the functionality below. The challenge is
              `model.fantasize(X)` adds an additonal dimension at the start of all posteriors calculated e.g.
              `(num_fantasies, batch_shape, n, m)`. Unclear if our QoI methods can handle/respect the `num_fantasies`
              dimension, of if the different fantasy models can easily be extracted. Revisit at a future date.
            - Consider making the jobs batchable or multiprocessed.
        """
        # shape is: (t_batch, d)
        x = X.squeeze(1)
        x_grad = x.requires_grad

        if x_grad:
            msg = (
                "Gradient tracking is not yet supported, please set `no_grad=True` in optimizer settings."
                "See tutorials/ax_botorch/botrch_minimal_example_custom_acq.ipynb for details"
            )
            raise NotImplementedError(msg)

        # .model.posterior(X) turns gradient back on unless torch.no_grad is used.
        with ExitStack() as stack:
            # sourcery skip: remove-redundant-if
            if not x_grad:
                stack.enter_context(torch.no_grad())

            reject_if_batched_model(
                self.model
            )  # the posterior predictions used in forward don't support batched-models
            # posterior = self.model.posterior(x)
            # shape is:  (n_posterior_samples, t_batch, m)
            # y_o = self.sampler(posterior)
            # print("y_o", y_o.shape)

            # shape is: (t_batch, m)
            # yvar_o = _get_fantasy_observation_noise(self.model, x)
            # print("yvar_o", yvar_o.shape)

            # print("x.shape", x.shape)
            param_order = list(search_space.parameters.keys())
            # print("param_order", param_order)
            # TODO: is the param_order and output_dist correct here?
            evaluation_function = EvaluationFunction(
                simulator=self.simulator,
                output_dist=dist,  # output_dist=posterior,
                parameter_order=param_order,
                n_simulations_per_point=200,  # x.shape[0],
            )
            param_dict = [{"x1": float(row[0]), "x2": float(row[1])} for row in x]
            # sim_point_results = evaluation_function(parameters=param_dict[0])
            sim_point_results = [evaluation_function(parameters=params) for params in param_dict]
            # print("sim_point_results", sim_point_results)
            loc_scale_means = [result.means for result in sim_point_results]
            y_loc_scale = torch.tensor(loc_scale_means).unsqueeze(0)
            # print("y_loc_scale", y_loc_scale.shape)
            y = y_loc_scale

            cov_matrices = [result.cov for result in sim_point_results]
            variances = [torch.tensor(cov.diagonal()) for cov in cov_matrices]
            yvar = torch.stack(variances)
            yvar_sqrt = torch.sqrt(yvar)
            yvar_squared = yvar**2
            yvar_from_se = yvar * np.sqrt(200)
            # yvar = _get_fantasy_observation_noise(self.model, x)

            # The below is the functionality of the _get_fantasy_observation_noise(self.model, x), originally used for
            #  yvar in QoILookAhead
            # Gaurenteeed to be (n,d)
            model_train_x = get_training_data_singletaskgp(model)
            # # Gaurenteeed to be (n,m) due to rejecting batched_models
            model_yvar = get_target_noise_singletaskgp(model)
            print("model_yvar", model_yvar.shape)

            # if model uses homoskedastic noise output will need expand output from (1,m) to (n,m)
            model_yvar = model_yvar.expand(model_train_x.shape[0], -1)

            x_obs_noise = average_observational_noise(x, model_train_x, model_yvar)

            # print("yvar[0]", yvar[0])
            # print("model_yvar[0]", model_yvar[0])
            print("yvar.shape", yvar.shape)
            print("model_yvar.shape", model_yvar.shape)
            # yvar = x_obs_noise  # model_yvar
            # print("sim_point_results[0]", sim_point_results[0])
            # print("yvar_sqrt[0]", yvar_sqrt[0])
            # print("yvar_squared[0]", yvar_squared[0])
            # print("yvar_o[0]", yvar_o[0])
            # print("yvar_from_se[0]", yvar_from_se[0])
            # print("yvar", yvar.shape)

            # yvar = yvar_squared
            # yvar = yvar_sqrt

            """
            # Get y from actual simulator
            print("self.simulator", self.simulator)
            print("x.shape", x.shape)
            y = self.simulator(x)
            print("y", y.shape)
            y = torch.tensor(y)
            print("y1", y.shape)
            y_flat = y.flatten()
            print("y_flat", y_flat.shape)
            """

            # extend the shapes so they all match (n_postior_samples, t_batch, m)
            x = x.expand(y.shape[0], -1, -1)
            yvar = yvar.expand(y.shape[0], -1, -1)

            # shape is: (n_posterior_samples, t_batch)
            lookahead_var = self._batch_lookahead(x, y, yvar)

            # the results of the different y samples need to be aggregated.
            # Use a bespoke aggregator if provided by the sampler, otherwise use the mean
            if hasattr(self.sampler, "mean"):  # noqa: SIM108
                lookahead_var = self.sampler.mean(dim=0)  # type: ignore
            else:
                lookahead_var = lookahead_var.mean(dim=0)

            # The output from an acquisition function is maximized, so we negate the QoI variances.
            lookahead_var = -lookahead_var

            if x_grad and not lookahead_var.requires_grad:
                msg = (
                    "Losing gradient in QoiLookAhead.forward()\n. Optimisation setting currently require gradient."
                    " This is likely due to gradient not propagated through the qoi_estimator."
                )
                raise RuntimeWarning(msg)

            return lookahead_var

    def _batch_lookahead(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        yvar: torch.Tensor,
    ) -> torch.Tensor:
        """Process a batch of lookahead points.

        Args:
            x (`*b`,n,d): x points to proceess
            y (`*b`,n,m): y points to process
            yvar (`*b`,n,m): yvar points to process

        Returns:
            torch.Tensor (`*b`,n) of lookahead results.
        """
        batch_shape = x.shape[:-2]
        n, d = x.shape[-2:]
        m = y.shape[-1]

        x_flat = x.reshape(-1, d)
        y_flat = y.reshape(-1, m)
        yvar_flat = yvar.reshape(-1, m)

        _results = []
        for x_point, y_point, yvar_point in zip(x_flat, y_flat, yvar_flat, strict=True):
            qoi_var = self.lookahead(x_point, y_point, yvar_point)
            _results.append(qoi_var)

        results = torch.tensor(_results)
        results = results.reshape(*batch_shape, n)
        return results

    def lookahead(self, x_point: torch.Tensor, y_point: torch.Tensor, yvar_point: torch.Tensor | None) -> torch.Tensor:
        """Performs a single lookahead calculation.

        Adds a single additional datapoint to the GP, and determines the QoI with the new GP.

        Args:
            x_point: (d,) The x location of the new point
            y_point: (m,) The y (target) of the new point
            yvar_point: (m,) The y_var  of the new point

        Returns:
            (,) Variance of the QoI with the lookahead GP
        """
        # if homoskedastic noise is used yvar needs to be None due to conditional_upate
        if isinstance(self.model.likelihood, GaussianLikelihood):
            yvar_point = None

        updated_model = conditional_update(
            self.model,
            X=x_point.unsqueeze(0),
            Y=y_point.unsqueeze(0),
            observation_noise=yvar_point.unsqueeze(0) if yvar_point is not None else None,
        )

        #  Calculate the QoI estimates with the fantasized model
        qoi_estimates = self.qoi(updated_model)

        # Calculate the variance of the QoI estimates
        return self.qoi.var(qoi_estimates)


def conditional_update(model: Model, X: torch.Tensor, Y: torch.Tensor, observation_noise: torch.Tensor | None) -> Model:  # noqa: N803
    """A wrapper around `BatchedMultiOutputGPyTorchModel.condition_on_observations` with a number of safety checks.

    This function adds an additional datapoint to the model, preserving the dimension of the original model. Does not
    changing any of the models hyperparameters. This is like training a new `SingleTaskGP` with all the datapoints
    (Hyperparameters are not fit in SingleTaskGP).

    Args:
        model (Model): The model to update.
        X: As per `condition_on_observations`. Shape (`*b`, n', d).

            .. Note::
                `condition_on_observations` expects this to be in the "model" space. It will not be
                transformed by the `input_transform` on the model.

        Y: As per `condition_on_observations`. Shape (`*b`, n', m).

            .. Note::
                `condition_on_observations` expects this to be in the "output/problem" space (not model space).
                It will be transformed by the `output_transform` on the model.

        observation_noise (torch.Tensor | None): Used as the `noise` argument in `condition_on_observations`. Shape
            should match `Y` (`*b`, n', m).

            .. Note::
                `condition_on_observations` expects this to be in the "model" space. It will not be
                transformed by the `output_transform` on the model.

    Return:
        - GP with the same underlying structure, including the new points, and the same original number of dimensions.

    Developer Note:
        There are different ways to create a fantasy model. The following were considered:

        - `BatchedMultiOutputGPyTorchModel.condition_on_observations`: well documented interface producing a GP of
          the same format.
        - `model.get_fantasy_model`: This is a `Gpytorch` implementation. Interface uses different notation, and
          input shape need to be manually adjusted depending on the model.
        - `model.fantasize`: This method would be very convient for our wider purpose, but its posteriors is of shape
          `(num_fantasies, batch_shape, n, m)`. Unclear if our QoI methods can handle/respect the `num_fantasies` dim.

        Revisit this at a later date.
    """
    if not isinstance(model, SingleTaskGP):
        msg = (
            f"Currently only supports models of type SingleTaskGP, received {type(model)}."
            "Currently this has only been tested for SingleTaskGP (with Homoskedastic and Fixed noise)."
            " It can likely be expanded to broader classes of model, but it is important to understand the noise"
            "expected by the model, as this can silently create incorrect GPs if incorrect. See TODO(link unit test)"
            " for details. As such, currently only GPs that have been explicitly tested are accepted."
        )
        raise NotImplementedError(msg)

    if isinstance(model.likelihood, FixedNoiseGaussianLikelihood):
        if observation_noise is None:
            msg = "Conditional update of SingleTaskGP with FixedNoise requires observation_noise to be provided."
            raise ValueError(msg)
    elif isinstance(model.likelihood, GaussianLikelihood):
        if observation_noise is not None:
            msg = (
                "Conditional update of the observation_noise is not supported for SingleTaskGPs with Homoskedastic "
                "Noise. This combination leads to inconsistent updated of the GP."
            )
            raise ValueError(msg)
    else:
        msg = (
            "Expected the model.likelihood to be either GaussianLikelihood or FixedNoiseGaussianLikelihood."
            " Recieved {model.likelihood}"
        )
        raise TypeError(msg)

    if hasattr(model, "input_transform") or hasattr(model, "outcome_transform"):
        msg = (
            "Caution: the model passed has input or outcome transforms. Need to be extremely careful that all inputs"
            "the in the correct space ('model space' vs 'problem/outcome space'). Please see docstring for more"
            " details."
        )
        warnings.warn(msg, stacklevel=5)

    if model.batch_shape != X.shape[:-2] or model.batch_shape != Y.shape[:-2]:
        msg = (
            f"Model.batch_shape is {model.batch_shape}, but X batch_shape is {X.shape[:-2]} and Y batch_shape is:"
            f" {Y.shape[:-2]}. The resultant model will not have the same batch shape as the original."
        )
        raise ValueError(msg)

    # Botorch required the model be have been run (so caches have been populated) before conditioning can be done.
    # This is the method Botrch used to do this. We run the model on a single input if required
    if model.prediction_strategy is None:
        _ = model.posterior(X=X)

    new_model = model.condition_on_observations(
        # MUST BE IN MODEL SPACE - will not be automatically transformed
        # shape require (b,n',d)
        X=X,
        # Must be in outcome space - will be automatically transformed
        # Shape require (b,n',m)
        Y=Y,
        # Must be in model space - - will not be automatically transformed
        # shape must be the same as Y
        noise=observation_noise,
    )

    return new_model


# TODO(sw 2024-11-27): there are cases where users might want to chose between closest_observational_noise and
# average_observational_noise. This should be a parameter of QoILookAhead.
def closest_observational_noise(
    new_points: torch.Tensor, train_x: torch.Tensor, train_yvar: torch.Tensor
) -> torch.Tensor:
    """Find the closest point in a training dataset, and collect its observational noise.

    Args:
        new_points: (n, d) The points to produce observational_noise. Features should be normalise to [0,1] square.
        train_x: (n',d) The points to compare similarity to. Features should be normalise to [0,1] square.
        train_yvar: (n',m) The observational variance associated with each point.

    Return:
        (n,m) Tensor with the variance for each of the new_points.


    Details:
    This function is useful for Non-Batched SingleTaskGPs because they will always have arguments of these dimension.

    Warning:
        This function is not smooth, meaning optimizers that use gradient (1st or 2nd order derivatives) such as
        L-BFGS-B will not work. The trade off is it is more robust to the effect of patterns in yvar than
        `average_observational_noise` . See Issue #213 for details.

    """
    # NOTE: Training data might be less than [0,1] because it can be standardised with preset bounds (e.g env bounds)
    # new_points could potentailly be outside of bounds if bounds were determined by the data.
    if train_x.min() < 0 or train_x.max() > 1:
        # TODO(sw 2024-11-15): We could use normalise to standardise the model here ourselves.
        msg = (
            "The models train_x data is not standardised to [0,1]. Non-standarised features will biases large"
            " features in the similarity measure used in this function."
            f" Found min={train_x.min()}, max={train_x.max()}."
        )
        warnings.warn(msg, stacklevel=8)

    # measure the distance from new_points to train_points to find the one that is most similar.
    # (new_point, distance to points)
    distances = torch.cdist(new_points, train_x, p=2)
    closest_point_idx = distances.argmin(dim=-1)
    return train_yvar[closest_point_idx]


def average_observational_noise(
    new_points: torch.Tensor,
    train_x: torch.Tensor | None,  # noqa: ARG001
    train_yvar: torch.Tensor,
) -> torch.Tensor:
    """Return the average observational noise.

    Args:
        new_points: (n, d) The points to produce observational_noise.
        train_x: (n',d). This is not used, but is kept to keep a consistent function signature
        train_yvar: (n',m) The observational variance associated with each point.

    Return:
        (n,m) Tensor with the variance for each of the new_points.

    Details:
    This function is useful for Non-Batched SingleTaskGPs because they will always have arguments of these dimension.

    Warning:
        Certain pattern of homoskedasticity cause this method to perform poorly, causing the acquisition function to
        recommend suboptimal points (as compared to `closest_observational_noise`). The trade off is derivative based
        optimisation techniques can be used. See Issue #213 for details.
    """
    return train_yvar.mean(dim=0).expand(new_points.shape[0], -1)


def _get_fantasy_observation_noise(model: SingleTaskGP, x: torch.Tensor) -> torch.Tensor:
    """Estimates the observation noise of a new point by finding the closest point in a model.

    Distance is assumed a good measure of point similarity.

    Args:
        model: The model from which to use the training data.
        x: (n,d) The points to find fantasy noise for. Batching is not supported. This should be in model space.

    Return:
        Tensor (n,m) of observation noise (variance) for each point in x.

    Todo TODO:
        - Review the implication noise picking choice on optimisation. See Issue #213 for details.
    """
    reject_if_batched_model(model)

    # Gaurenteeed to be (n,d)
    model_train_x = get_training_data_singletaskgp(model)
    # # Gaurenteeed to be (n,m) due to rejecting batched_models
    model_yvar = get_target_noise_singletaskgp(model)

    # if model uses homoskedastic noise output will need expand output from (1,m) to (n,m)
    model_yvar = model_yvar.expand(model_train_x.shape[0], -1)

    x_obs_noise = average_observational_noise(x, model_train_x, model_yvar)

    return x_obs_noise


def reject_if_batched_model(model: SingleTaskGP) -> None:
    """Helper function to reject batched model in code where they are not yet supported.

    Args:
        model: The model to check

    Return:
        Raise not yet implements in model is batched. Otherwise None.

    Details:
        botorch models can have batched training data, and or batched.

        - gp batch prediction (non-batched model):

          - train_x = (n,d) # This is a single GP
          - train_y = (n,m)
          - predicting_x = (b,n',d)
          - result will be: (b, n',m).

            - There are b seperate joint distribution (each with n points, a t targets)

        - batched gps model:

          - train_x = (b_gp,n,d)
          - train_y = (b_gp,n,m)

            - b_gp seperate GPs, where each GP gets all its own hyperparams etc trained on (n,d) point.

          - prediciton_x = (n',d)
          - result will be: (b_gp, n',m).

            - Each of the seperate b_gp gps makes its own estimate of the joint distribution.

        More details: `BoTorch batching <https://botorch.org/docs/batching>`_
    """
    if len(model.batch_shape) != 0:
        msg = (
            "batch models are not currently supported. Model has training data (*b,n,d), on (n,d) is supported."
            "See https://botorch.org/docs/batching for more details on batching"
        )
        raise NotImplementedError(msg)


@acqf_input_constructor(OptimalLookAhead)
def construct_inputs_sim_qoi(
    model: SingleTaskGP,
    qoi_estimator: QoIEstimator,
    simulator: Simulator,
    sampler: PosteriorSampler,
    **_: dict[str, Any],
) -> dict[str, Any]:
    """This is how default arguments for acquisition functions are handled in Ax/Botorch.

    Context
        When Ax.BOTORCH gets instantiated, construction arguments for the acquisition function can be provided.
        These are passed through Ax as a set of Kwargs

    Args:
        This function takes a subset of the acquisition functions `__init__()` args and can add defaults.

    Returns:
        Args for the Botorch acquisition function `__init__()` (output).

    Note:
        This functionality allows Ax to pass generic arguments without needing to know which acquisition function they
        will be passed to. Interestingly, this functionality is provided by the BoTorch package, even though it seems
        like it should be the responsibility of Ax. This issue is discussed in detail here: `GitHub discussion <https://github.com/pytorch/botorch/discussions/784>`_.
    """
    return {
        "model": model,
        "qoi_estimator": qoi_estimator,
        "simulator": simulator,
        "sampler": sampler,
    }


@optimizer_argparse.register(OptimalLookAhead)
def _argparse_qoi_look_ahead(
    # NOTE: this is a bit of a non-standard implementation because we need to update params in a nested dict. Would
    #  prefer to set the key values directly here
    acqf: OptimalLookAhead,
    # needs to accept the variety of args it is handed, and then pick the relevant ones
    **kwargs: Any,  # noqa: ANN401
) -> dict[str, Any]:
    """Controls the default optimization parameters used.

    The following provides an overview of how passing optimization argument to `ax.TorchModelBridge.gen()`.
    Anything that is not passed here will used the default value defined by this function.

    ```python
        TorchModelBridge.gen(
            ...                                 # Other parameters there that are not relevant
            model_gen_options: {
                optimise_kwargs: {              # content of this key later used like `botorch.optim.optimize_acqf
                (**optimise_kwargs)`  # noqa: E501
                    options: {                  # This is a parameter of `botorch.optim.optimize_acqf`
                        with_grad: False        # Example of some optimization params that can be passed
                        ...
                        method: "L-BFSG"
                    }
                }
            }
    ```

    Rough flow of this object through the stack:
        - `ax.TorchModelBridge.gen(..., model_gen_options: dict)`
        [internally end up calling `TorchModelBridge.model` giving the below object]
        - `ax.BotorchModel.gen(..., torch_opt_config: TorchOptimConfig)`
            - Same way TorchModelBridge convert x data from ax type to tensors, the config type is converted.
        [Internally that leads to the following being called]
        - `ax.Acquisition.optimise(..., opt_options)`
            - where `opt_options = torch_opt_config.model_gen_options.optimize_kwargs`
        [Internally this leads to the following being called]
        - `botorch.optim.optimize_acqf(..., opt_options_with_defaults)`
            - `opt_options_with_defaults` is `opt_options` with defaults added.

    This function helps add the defaults in the final step

    Note:
        - This function is stored in a registry in Ax. This registery is searched (by acquisition class name)
          when the acqf func is being optimized.
        - Any variable passed from `.gen` must override the defaults specified here
        - Existing default implementations can be found here: ax.models.torch.botorch_modular.optimizer_argparse

    Args:
        acqf: Acqusiton that will be passed.
        kwargs:
            - everything passed in value of `optimizer_kwargs` in `model.gen(model_gen_options =
            {optimizer_kwargs ={}})`
            can be found with
            `kwargs[optimizer_options]`

    Returns:
        dict of args unpacked to `botorch.optim.optimize_acqf`.
            - All args are set with this, excluding the following:
                - `acq_function`,`bounds`,`q`,`inequality_constraints`,`fixed_features`,`post_processing_func`
    """
    # Start by using the default arg constructure, then adding in any kwargs that were passed in.

    # NOTE: this is an internal method, using it is an anti pattern.
    # - Ax just stores all the arge parsers in the same file as _argparse_base so they can use it directly.
    # - We can't put this function in that file, even though it belongs there.
    # Definition of _argparse_base explains the shape returned
    args = _argparse_base(acqf, **kwargs)

    # Only update with these defaults if the variable were not passin in kwargs
    optimizer_options = kwargs["optimizer_options"]
    options = optimizer_options.get("options", {})

    if "raw_samples" not in optimizer_options:
        args["raw_samples"] = 100
    if "with_grad" not in options:
        args["options"]["with_grad"] = False

    if "method" not in options:
        args["options"]["method"] = "Nelder-Mead"
        # These options are specific to using Nelder-Mead
        if "maxfev" not in options:
            args["options"]["maxfev"] = "5"
        if "retry_on_optimization_warning" not in optimizer_options:
            args["retry_on_optimization_warning"] = False

    return args


# %%
acqf = OptimalLookAhead
posterior_sampler = sampling.MeanSampler()


def look_ahead_sim_generator_run(experiment: Experiment) -> GeneratorRun:
    # Fist building model to get the transforms
    # TODO (se -2024-11-20): This refits hyperparameter each time, we don't want to do this.
    model_bridge_only_model = Models.BOTORCH_MODULAR(
        experiment=experiment,
        data=experiment.fetch_data(),
    )
    input_transform, outcome_transform = transforms.ax_to_botorch_transform_input_output(
        transforms=list(model_bridge_only_model.transforms.values()), outcome_names=model_bridge_only_model.outcomes
    )
    # Adding the transforms to the QoI estimator
    qoi_estimator.input_transform = input_transform
    qoi_estimator.outcome_transform = outcome_transform

    # Building the model with the QoILookAhead acquisition function
    model_bridge_cust_ac = Models.BOTORCH_MODULAR(
        experiment=experiment,
        data=experiment.fetch_data(),
        botorch_acqf_class=acqf,
        acquisition_options={
            "qoi_estimator": qoi_estimator,
            "simulator": sim,
            "sampler": None,  # "sampler": posterior_sampler,
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

exp_sim_qoi = make_exp()
# This needs to be instantiated outside of the loop so the internal state of the generator persists.
sobol = Models.SOBOL(search_space=exp_sim_qoi.search_space, seed=5)


def sobol_generator_run(_: Experiment) -> GeneratorRun:  # pyright: ignore[reportRedeclaration]
    return sobol.gen(1)


qoi_results_look_ahead_sim = run_trials(
    experiment=exp_sim_qoi,
    warm_up_generator=sobol_generator_run,
    doe_generator=look_ahead_sim_generator_run,
    qoi_estimator=qoi_estimator,
    warm_up_runs=warm_up_runs,
    doe_runs=n_iter,
    qoi_iter=qoi_iter,
)

# %%
print("qoi_results_look_ahead_sim", qoi_results_look_ahead_sim.shape)
mean, var = get_mean_var(qoi_estimator, qoi_results_look_ahead)
mean_bf, var_bf = get_mean_var(qoi_estimator, qoi_results_look_ahead_sim)
mean_bf1 = qoi_results_look_ahead_sim.mean(axis=1)
var_bf1 = qoi_results_look_ahead_sim.var(axis=1)
print("mean, var", mean, var)
print("mean_bf, var_bf", mean_bf, var_bf)
print(mean_bf1.shape, var_bf1.shape)
print(mean_bf.shape, var_bf.shape)
print("mean_bf1, var_bf1[-1]", mean_bf1[-1], var_bf1[-1])
print("mean_bf[-1]", mean_bf[-1])
# %%
# Plot the QoI estimates and their uncertainty for the Sobol, Look Ahead methods and Bruteforce Sim method
mean, var = get_mean_var(qoi_estimator, qoi_results_sobol)
ax = plot_raw_ut_estimates(mean, var, name="Sobol")
mean, var = get_mean_var(qoi_estimator, qoi_results_look_ahead)
ax = plot_raw_ut_estimates(mean, var, ax=ax, color="green", name="look ahead")
mean_bf, var_bf = get_mean_var(qoi_estimator, qoi_results_look_ahead_sim)
ax = plot_raw_ut_estimates(mean_bf, var_bf, ax=ax, color="red", name="brute force")
# ax = plot_raw_ut_estimates(mean_bf1, var_bf1, ax=ax, color="red", name="brute force")
_ = ax.axhline(brute_force_qoi_estimate, c="black", label="brute_force_value")
_ = ax.set_xlabel("Number of DOE iterations")
_ = ax.set_ylabel("Response")
_ = ax.legend()

# %% How long does a single run take
acqusition = OptimalLookAhead(model, qoi_estimator, sim)
scores = acqusition(torch.tensor([[[0.5, 0.5]]]))
print("scores", scores)

# %% Perform the grid search and plot
point_per_dim = 21
x1 = torch.linspace(0, 1, point_per_dim)
x2 = torch.linspace(0, 1, point_per_dim)
grid_x1, grid_x2 = torch.meshgrid(x1, x2, indexing="xy")
grid = torch.stack([grid_x1, grid_x2], dim=-1)
# make turn into a shape that can be processsed by the acquisition function
x_candidates = grid.reshape(-1, 1, 2)
# acqusition = QoILookAhead(model, qoi_estimator)
scores = acqusition(x_candidates)
scores = scores.reshape(grid.shape[:-1])

# %%
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")
_ = ax.view_init(elev=30, azim=45)  # pyright: ignore[reportAttributeAccessIssue]
_ = ax.plot_surface(grid_x1, grid_x2, scores, cmap="viridis", edgecolor="none")  # pyright: ignore[reportAttributeAccessIssue]
_ = ax.set_xlabel("x1")
_ = ax.set_ylabel("x2")
_ = ax.set_zlabel("score")  # pyright: ignore[reportAttributeAccessIssue]
_ = ax.set_title("Score surface plot OptimalLookAhead")
print("max_score ", scores.max())


# %%
