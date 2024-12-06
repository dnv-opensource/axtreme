"""Simplified end-to-end example of using axtreme to solve a toy problem.

The following creates a toy problem, and calculates the brute force solution. We then demonstrate how axtreme can be
used to achieve the same results while running the simulator far fewer times. Specially, we show how to:
- Define the problem in the Ax framework.
- Create a surrogate model (automatically using ax).
- Estimate the QoI.
- Use DoE to reduce uncertainty.

NOTE: This is an introductory example intended to provide an overview of the process. As such, a number of
simplification are made. More indepth tutorial are provided here (TODO(sw 2024-11-23): provide a link).
"""
# pyright: reportUnnecessaryTypeIgnoreComment=false

# %%
import sys
from collections.abc import Callable
from functools import partial
from pathlib import Path

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
from numpy.typing import NDArray
from plotly.subplots import make_subplots
from scipy.stats import gumbel_r
from torch.utils.data import DataLoader

from axtreme import sampling
from axtreme.acquisition import QoILookAhead
from axtreme.data import BatchInvariantSampler2d, FixedRandomSampler, MinimalDataset
from axtreme.experiment import add_sobol_points_to_experiment, make_experiment
from axtreme.plotting.doe import plot_qoi_estimates
from axtreme.plotting.gp_fit import plot_gp_fits_2d_surface, plot_surface_over_2d_search_space
from axtreme.plotting.histogram3d import histogram_surface3d
from axtreme.qoi import GPBruteForce
from axtreme.sampling import NormalIndependentSampler
from axtreme.simulator import utils as sim_utils
from axtreme.utils import population_estimators, transforms

torch.set_default_dtype(torch.float64)
device = "cpu"

# %%
#################### Explore the mock problem being solved #############################
########################################################################################
"""The axtreme package expects 2 key peices of input:
- A simulator:
- Samples from the environment

Here we use a mock simulator and env data defined in `exmaples`. The following section explores these raw inputs.
"""
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))
from examples.demo2d.problem.env_data import collect_data
from examples.demo2d.problem.simulator import _true_loc_func, _true_scale_func, dummy_simulator_function

# %%
""" Inputs: The Simulatore
For a given x, the simulator produces a noisy y output.

The toy simulator in this example uses a gumbel distribution as its noise model. A gumbel distribtuion has 2 parameters,
 location (loc) and scale. The toy problem has one underling function that controls the loc, and another to control the
 scale. This can be written os:
 - y = loc(x) + noise, where noise ~ Gumbel(0, scale(x))
 OR
 - y = Gumbel(loc(x), scale(x)).sample()

In a real problem the underlying function that control the output distribution would be unknown, but in this toy example
we plot them directly to give a better understanding of the problem being solved in this example.
"""
# plot it
fig = make_subplots(
    rows=1,
    cols=3,
    specs=[[{"type": "surface"}, {"type": "surface"}, {"type": "surface"}]],
    subplot_titles=("location", "scale", "Gumble response surface (at q = [.1, .5, .9]"),
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
def gumbel_helper(x: np.ndarray[tuple[int, int], np.dtype[np.float64]], q: float = 0.5) -> NDArray[np.float64]:  # noqa: D103
    return gumbel_r.ppf(q=q, loc=_true_loc_func(x), scale=_true_scale_func(x))


quantile_plotters = [partial(gumbel_helper, q=q) for q in [0.1, 0.5, 0.9]]
response_distribution = plot_surface_over_2d_search_space(plot_search_space, funcs=quantile_plotters)
_ = [fig.add_trace(data, row=1, col=3) for data in response_distribution.data]

# label the plot
_ = fig.update_scenes({"xaxis": {"title": "x1"}, "yaxis": {"title": "x2"}, "zaxis": {"title": "response"}})
_ = fig.update_traces(showscale=False)
fig.show()


# %%
"""Inputs: The Environment data
The other input required by the axtreme package is static dataset of environment samples.
Here we load and plot the data we will be using in this example.
"""
import pandas as pd

raw_data: pd.DataFrame = collect_data()
display(raw_data.head(3))  # type: ignore  # noqa: F821, PGH003

# we convert this data to Numpy for ease of use from here on out
env_data: NDArray[np.float64] = raw_data.to_numpy()
fig = histogram_surface3d(env_data)
_ = fig.update_layout(title_text="Environment distribution estimate from samples")
_ = fig.update_layout(scene_aspectmode="cube")
fig.show()

# %%
"""Define the Problem:
Because we are using a Toy example we can directly calculate the ERD and our QOI. This is the answer that we are trying
to recover using the axtreme package (in which we make minimal use of the simulator).

We define the time span overwhich we wish to calculate the Extreme response, and produce a brute force estiamte.
"""
# define the timespane
N_ENV_SAMPLES_PER_PERIOD = 1000

# %%
n_erd_samples = 1_000
erd_samples = []
for _ in range(n_erd_samples):
    indices = np.random.choice(env_data.shape[0], size=N_ENV_SAMPLES_PER_PERIOD, replace=True)  # noqa: NPY002
    period_sample = env_data[indices]

    responses = dummy_simulator_function(period_sample)
    erd_samples.append(responses.max())


# %%
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

# %%
brute_force_qoi_estimate = np.median(erd_samples)
print(f"Brute force estimate of our QOI is {brute_force_qoi_estimate}")


# %%
"""Solving the problem with Rax.

In the section above we have explored the key inputs to the axtreme package (Simulator and Environment data), and
calculated the brute force answer to our QoI. In this section we will show how Rax can be used to achieve the same
result, while running the simulator far fewer times.

Rax is comprised of 3 main step:
- Define the problem in the Ax framework:
- Create a surrogate model: (this happens automatically once the problem has been defined in Ax).
- Estimate the Qoi: Estimate the QoI (and our confidence) using the surrogate model.
- DoE: reduce uncertainty in the QoI while running the simulator as little as possible.
"""

############################# Put Problem in AX framework ##############################
########################################################################################

# %%
""" Define the problem in the Ax framework.

The following steps need to be taken to define the problem in the Ax framework:
- Ensure the simulator conforms to the required interface.
- Decide on the search space to use.
- Pick a distribution that you believe captures the noise behaviour of your simulator.

These decisions are straight forward for this toy example. Advice on how to choose these parameters in more real world
problems are provided in other tutorials (TODO(sw 2024-11-21): include these tutorials).
"""
### Make our simulator conform to the required interface
# Check if it complies
print(sim_utils.is_valid_simulator(dummy_simulator_function, verbose=True))
# use a helper to add the 'n_simulations_per_point' funcitonslity
sim = sim_utils.simulator_from_func(dummy_simulator_function)
# check that it now complies
print(sim_utils.is_valid_simulator(sim, verbose=True))

# %%
### Pick the search space over which to create a surrogate
search_space = SearchSpace(
    parameters=[
        RangeParameter(name="x1", parameter_type=ParameterType.FLOAT, lower=0, upper=1),
        RangeParameter(name="x2", parameter_type=ParameterType.FLOAT, lower=0, upper=1),
    ]
)
# %%
### Pick a distibution that you belive captures the noise behvaiour of your simulator
dist = gumbel_r


# %%
### Automatically set up you experiment using the sim, search_space, and dist defined above.
def make_exp() -> Experiment:
    """Helper to ensure we always create an experiment with the same settings (so results are comparable)."""
    return make_experiment(sim, search_space, dist, n_simulations_per_point=25)


exp = make_exp()

# %%
""" Create a Surrogate model:

Once we have defined an Ax experiment, we can use Ax to create a surrogate.This surrogate provides the location and
scale parameters for our gumbel distribution.

We first need to add some training data to the experiement. We do this by passing the experiement some inital x points.
Behind the scenes, Ax will run the Simulator to obtain estimates for the location and scale parameters at those points.
It will then use that data to create a surrogate model.
"""
# Add random x points to the experiment
add_sobol_points_to_experiment(exp, n_iter=30, seed=8)

# Use ax to generate a surroagate
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
_ = plt.plot(x_points, pred_dist.pdf(x_points), label="Surrogate")  # type: ignore  # noqa: PGH003
_ = plt.xlabel("Response value")
_ = plt.ylabel("pdf")
_ = plt.title("Surrogate distribution vs Simulator distribution at point x = [0.5, 0.5]")
_ = plt.legend()
plt.show()

# %%
# The surrogate also contians uncertainty about its estimate. Lets plot the other distributions that the surrogate
# believes are possible.
# TODO(sw): was there some helper for doing this more easily?


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
surrogate_distribution_samples = surrogate_distribution.rvs(size=5)
for sample in surrogate_distribution_samples:
    sample_dist = dist(loc=sample[0], scale=sample[1])
    _ = plt.plot(x_points, sample_dist.pdf(x_points), c="grey", alpha=0.5)  # type: ignore  # noqa: PGH003

_ = plt.plot(x_points, pred_dist.pdf(x_points), c="orange", label="Surrogate")  # type: ignore  # noqa: PGH003
_ = plt.title("Distributiion of possible responses at x = [0.5, 0.5]")
_ = plt.xlabel("Response value")
_ = plt.ylabel("pdf")
_ = plt.legend()
plt.show()
# %%
""" Estimate the QoI:
Now that we have a surroage model, we can use it to estimate the QoI. The uncertainty in our surrogate model should be
reflected in our QoI estimate. Lets demonstrate how the estiamte chages as we add more data to the surrogate model and
it becomes more certain.

In the following we make use of an existing QoI Estimator. It works in a very similar way to the brute force estimate,
except it uses the surrogate model to make predictions, rather than the simulator. Rax provides a number of
QoIEstimators for common tasks, but users can also create custom QoIEstimator for their specific problems. Details can
be found #TODO(sw 2024-11-22): link to tutorial on how to create a custom QoIEstimator).

In the following we demonstrate how the QoI estimate becomes more certain as the surrogate gets more training data.

NOTE: Training a Gp with `Models.BOTORCH_MODULAR` has inherit randomness that can't be turned off (e.g with
`torch.manual_seed`). As a result there is slight randomness in the result even though all other seeds are set.
"""
# A note on DataLoaders:
# We make use of Dataloader to manage providing data to the QoI. Env samples are only used in the QoI, so your env data
# can be in whatever format is supported by the QoI.
n_periods = 3  # 21 seems goo for this example
dataset = MinimalDataset(env_data)

sampler = FixedRandomSampler(dataset, num_samples=n_periods * N_ENV_SAMPLES_PER_PERIOD, seed=10, replacement=True)

batch_sampler = BatchInvariantSampler2d(
    sampler=sampler,
    batch_shape=torch.Size([n_periods, 256]),  # 64 seems good for this example
)
dataloader = DataLoader(dataset, batch_sampler=batch_sampler)

qoi_estimator_gp_brute_force = GPBruteForce(
    env_iterable=dataloader,
    erd_samples_per_period=1,
    posterior_sampler=NormalIndependentSampler(torch.Size([10]), seed=14),  # 50 and 14 seems good for this example
    # When true, All uncertainty shown in estimates is due to uncertainty in the surrogate model.
    shared_surrogate_base_samples=True,
    no_grad=True,
    seed=12,
)

# %%
### Intialise and environement, get gp from ax, and calculate the QOI
n_training_points = [40, 64, 128, 1024]
results = []

for points in n_training_points:
    exp = make_experiment(sim, search_space, dist, n_simulations_per_point=100)
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
    qoi_estimator_gp_brute_force.input_transform = input_transform
    qoi_estimator_gp_brute_force.outcome_transform = outcome_transform

    model = botorch_model_bridge.model.surrogate.model
    # reseed the dataloader each time so the dame dataset is used.
    results.append(qoi_estimator_gp_brute_force(model))


# %%
_, axes = plt.subplots(nrows=len(n_training_points), sharex=True, figsize=(6, 6 * len(n_training_points)))

for ax, estimate, n_points in zip(axes, results, n_training_points, strict=True):
    ax.hist(estimate, density=True, bins=len(estimate) // 3)  # Plot the QoI estimates
    ax.axvline(brute_force_qoi_estimate, c="orange", label="Brute force results")
    _ = population_estimators.plot_dist(population_median_est_dist, ax=ax, c="orange", label="QOI estimate")
    ax.set_title(f"QoI esstimate with {n_points} training points")
    ax.set_xlabel("Response")
    ax.set_ylabel("Density")
    ax.legend()

# Note: if you are interested, the following allows you to see the result of the QoIEstimator if uncertainty is ignored.
# this may not be a very good estimate if the simulator mean doesn't match the surroage.
# from axtreme.eval.qoi import qoi_ignoring_gp_uncertainty  # noqa: ERA001
# resutl_no_gp_uncertainty = qoi_ignoring_gp_uncertainty(qoi_estimator_gp_brute_force, model)  # noqa: ERA001

# %%
""" DoE: effeciently reduce uncertainty in the QoI.

In the plots above we see QoI uncertainty can be reduced by adding training data. In this section we explore if we can
obtain a similar reduction in uncertainty with fewer training point. We do this by picking new points intelligently.

First we perform DoE using a space filling design (Sobol). This is the baseline that DoE should be able to imporve on.
We then run DoE using our custom acquistion function (QoILookAhead), and compare results.
"""


def run_trials(
    experiment: Experiment,
    warm_up_generator: Callable[[Experiment], GeneratorRun],
    doe_generator: Callable[[Experiment], GeneratorRun],
    qoi_estimator: GPBruteForce,
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
        if i in (0, n_iter):
            figs.append(
                plot_gp_fits_2d_surface(model_bridge, search_space, {"loc": _true_loc_func, "scale": _true_scale_func})
            )
        print(f"iter {i} done")

    for fig in figs:
        fig.show()
    return np.vstack(qoi_results)


# %%
## How many iterations to run in the following DOEs
n_iter = 10
qoi_iter = 1
warm_up_runs = 40

# %%
### Surrogate trained without acquisiton function for comparision
exp_sobol = make_exp()
# This needs to be instantiated outside of the loop so the internal state of the generator persists.
sobol = Models.SOBOL(search_space=exp_sobol.search_space, seed=5)


def sobol_generator_run(_: Experiment) -> GeneratorRun:  # noqa: D103 # pyright: ignore[reportRedeclaration]
    return sobol.gen(1)


qoi_results_sobol = run_trials(
    experiment=exp_sobol,
    warm_up_generator=sobol_generator_run,
    doe_generator=sobol_generator_run,
    qoi_estimator=qoi_estimator_gp_brute_force,
    warm_up_runs=warm_up_runs,
    doe_runs=n_iter,
    qoi_iter=qoi_iter,
)

# %%
""" Cusom acqusition function:
We now set up the components required for the Custom acquistion function.

NOTE: It is important to understand the behaviour of the acquistion function, as this has implication for the
optimisation arguements used. The parameters set here are a robust option that do not assume a smooth function.
If you can gaurentee the acquistion function is smooth this should be updated to use more effeceint methods.
"""
# %%
"""Optional: Explore optimisation settings

Prior to doing multiple rounds of optimisation, it can be useful to check the optimisation setting are appropriate
for the acquisition function. Here we perform and plot a grid search, and compare it to the optimisation value to get a
sense of the surface. This is more challenging to do in real problems

TODO(sw 2024-11-29): Clean this up and move into a util function
"""
# Define the mdoel to use.
exp = make_experiment(sim, search_space, dist, n_simulations_per_point=100)
add_sobol_points_to_experiment(exp, n_iter=128, seed=8)

# Use ax to create a gp from the experiment
botorch_model_bridge = Models.BOTORCH_MODULAR(
    experiment=exp,
    data=exp.fetch_data(),
)

# We need to collect the transforms used to the model gives result in the problem/outcome space.
input_transform, outcome_transform = transforms.ax_to_botorch_transform_input_output(
    transforms=list(botorch_model_bridge.transforms.values()), outcome_names=botorch_model_bridge.outcomes
)
qoi_estimator_gp_brute_force.input_transform = input_transform
qoi_estimator_gp_brute_force.outcome_transform = outcome_transform

model = botorch_model_bridge.model.surrogate.model

# %% Perform the grid search and plot
point_per_dim = 21
x1 = torch.linspace(0, 1, point_per_dim)
x2 = torch.linspace(0, 1, point_per_dim)
grid_x1, grid_x2 = torch.meshgrid(x1, x2, indexing="xy")
grid = torch.stack([grid_x1, grid_x2], dim=-1)
# make turn into a shape that can be processsed by the acquistion function
x_candidates = grid.reshape(-1, 1, 2)
acqusition = QoILookAhead(model, qoi_estimator_gp_brute_force)
scores = acqusition(x_candidates)
scores = scores.reshape(grid.shape[:-1])

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")
_ = ax.plot_surface(grid_x1, grid_x2, scores, cmap="viridis", edgecolor="none")  # pyright: ignore[reportAttributeAccessIssue]
_ = ax.set_xlabel("x1")
_ = ax.set_ylabel("x2")
_ = ax.set_zlabel("score")  # pyright: ignore[reportAttributeAccessIssue]
_ = ax.set_title("Score surface plot")
print("max_score ", scores.max())

# %% perform a round of optimisation using the under the hood optimiser
candidate, result = optimize_acqf(
    acqusition,
    bounds=torch.tensor([[0.0, 0.0], [1.0, 1.0]]),
    q=1,
    num_restarts=20,
    raw_samples=200,
    options={
        "with_grad": False,  # True by default.
        "method": "Nelder-Mead",  # "L-BFGS-B" by default
        "maxfev": 5,
    },
    retry_on_optimization_warning=False,
)
print(candidate, result)

# %%
""" Lookahead acquisition function
Define the helpers to run the cusom acqusition function.
"""
acqf_class = QoILookAhead
exp_look_ahead = make_exp()
posterior_sampler = sampling.MeanSampler()


def look_ahead_generator_run(experiment: Experiment) -> GeneratorRun:  # noqa: D103
    # Fist building model to get the transforms
    # TODO (se -2024-11-20): This refits hyperparameter each time, we don't want to do this.
    model_bridge_only_model = Models.BOTORCH_MODULAR(
        experiment=experiment,
        data=experiment.fetch_data(),
    )
    input_transform, outcome_transform = transforms.ax_to_botorch_transform_input_output(
        transforms=list(model_bridge_only_model.transforms.values()), outcome_names=model_bridge_only_model.outcomes
    )
    # Using lower fidelity verison of qoi estimate for the look ahead optimization
    # Adding the transforms to the QoI estimator
    qoi_estimator_gp_brute_force.input_transform = input_transform
    qoi_estimator_gp_brute_force.outcome_transform = outcome_transform

    # Building the model with the QoILookAhead acquisition function
    model_bridge_cust_ac = Models.BOTORCH_MODULAR(
        experiment=experiment,
        data=experiment.fetch_data(),
        botorch_acqf_class=acqf_class,
        acquisition_options={
            "qoi_estimator": qoi_estimator_gp_brute_force,
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
                "raw_samples": 100,
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
exp_gp_bruteforce = make_exp()
# This needs to be instantiated outside of the loop so the internal state of the generator persists.
sobol = Models.SOBOL(search_space=exp_gp_bruteforce.search_space, seed=5)


def sobol_generator_run(_: Experiment) -> GeneratorRun:  # noqa: D103
    return sobol.gen(1)


qoi_results_look_ahead = run_trials(
    experiment=exp_gp_bruteforce,
    warm_up_generator=sobol_generator_run,
    doe_generator=look_ahead_generator_run,
    qoi_estimator=qoi_estimator_gp_brute_force,
    warm_up_runs=warm_up_runs,
    doe_runs=n_iter,
    qoi_iter=qoi_iter,
)
# %%

ax = plot_qoi_estimates(qoi_results_sobol, name="Sobol")
ax = plot_qoi_estimates(qoi_results_look_ahead, ax=ax, color="green", name="look ahead")
_ = ax.axhline(brute_force_qoi_estimate, c="black", label="brute_force_value")
_ = ax.set_xlabel("Number of DOE iterations")
_ = ax.set_ylabel("Response")
_ = ax.legend()

# %%
