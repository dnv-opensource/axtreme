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
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import scipy
import torch
from ax import (
    Experiment,
    SearchSpace,
)
from ax.core import GeneratorRun, ObservationFeatures, ParameterType, RangeParameter
from ax.modelbridge import ModelBridge
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
from axtreme.metrics import QoIMetric
from axtreme.plotting.doe import plot_qoi_estimates_from_experiment
from axtreme.plotting.gp_fit import plot_gp_fits_2d_surface_from_experiment, plot_surface_over_2d_search_space
from axtreme.plotting.histogram3d import histogram_surface3d
from axtreme.qoi import MarginalCDFExtrapolation
from axtreme.qoi.qoi_estimator import QoIEstimator
from axtreme.sampling.ut_sampler import UTSampler
from axtreme.simulator import utils as sim_utils
from axtreme.utils import population_estimators, transforms

# Set a useful default view angle for #D plots
pio.templates["plotly"].layout.scene.camera.eye = {"x": 0.6, "y": -1.75, "z": 0.8}  # pyright: ignore[reportAttributeAccessIssue, reportOptionalMemberAccess]

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
fig_true_sim = make_subplots(
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
_ = fig_true_sim.add_trace(
    plot_surface_over_2d_search_space(plot_search_space, funcs=[_true_loc_func]).data[0], row=1, col=1
)
_ = fig_true_sim.add_trace(
    plot_surface_over_2d_search_space(plot_search_space, funcs=[_true_scale_func]).data[0], row=1, col=2
)


# Plot the response surface at different quantiles
def gumbel_helper(x: np.ndarray[tuple[int, int], np.dtype[np.float64]], q: float = 0.5) -> NDArray[np.float64]:
    return gumbel_r.ppf(q=q, loc=_true_loc_func(x), scale=_true_scale_func(x))


quantile_plotters: list[Callable[[np.ndarray[Any, np.dtype[np.float64]]], np.ndarray[Any, np.dtype[np.float64]]]] = [
    partial(gumbel_helper, q=q) for q in [0.1, 0.5, 0.9]
]
response_distribution = plot_surface_over_2d_search_space(plot_search_space, funcs=quantile_plotters)
_ = [fig_true_sim.add_trace(data, row=1, col=3) for data in response_distribution.data]

# label the plot
_ = fig_true_sim.update_scenes({"xaxis": {"title": "x1"}, "yaxis": {"title": "x2"}, "zaxis": {"title": "response"}})
_ = fig_true_sim.update_traces(showscale=False)
fig_true_sim.show()


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
precalced_erd_samples, precalced_erd_x = collect_or_calculate_results(N_ENV_SAMPLES_PER_PERIOD, 300_000)
brute_force_qoi_estimate = np.median(precalced_erd_samples)
print(f"Brute force estimate of our QOI is {brute_force_qoi_estimate}")

# %% [markdown]
## Brute Force Extreme Response Locations
# Additionally, we have precalculated the location of the extreme responses. For this toy problem this can be useful
# in order to understand both the problem and how `axtreme` works to solve such problems.
# Below we plot the extreme response locations.

# %%
# Create combined histogram of environment data and extreme responses
fig_combined = go.Figure()

# Add environment data traces
fig_env = histogram_surface3d(env_data)
_ = [
    fig_combined.add_trace(trace.update(colorscale="Blues", name="Environment Data", showscale=False))  # type: ignore  # noqa: PGH003
    for trace in fig_env.data
]

# Add extreme responses traces
fig_extreme = histogram_surface3d(precalced_erd_x.numpy())
_ = [
    fig_combined.add_trace(trace.update(colorscale="Reds", name="Extreme Responses", showscale=False))  # type: ignore  # noqa: PGH003
    for trace in fig_extreme.data
]

_ = fig_combined.update_layout(title_text="Environment Data(blue) vs Extreme Responses(red)")


fig_combined.show()


# %% Plot extreme response location on simulator location and scale surfaces
# Helper to Plot the extreme response locations on the underlying location and scale that control the simulator.
class Histogram2DLookup:
    """
    A helper which learns a histogram from a 2d dataset, and allows you to query the histogram value at a point.
    """

    def __init__(
        self,
        data: NDArray[Any],
        x_bounds: tuple[float, float] = (0.0, 1.0),
        y_bounds: tuple[float, float] = (0.0, 1.0),
        bins: int = 50,
    ) -> None:
        """
        Args;
            data: (n,2) array of points
            x_bounds: (min, max) bounds for the first column of data
            y_bounds: (min, max) bounds for the second column of data
            bins: number of bins each dimension should be split into.
        """
        # Creates a 2d matrix counting the occurances that fall in that bin
        self.hist, self.x_edges, self.y_edges = np.histogram2d(
            data[:, 0],
            data[:, 1],
            bins=bins,
            range=[x_bounds, y_bounds],
        )
        self.x_bin_width = self.x_edges[1] - self.x_edges[0]
        self.y_bin_width = self.y_edges[1] - self.y_edges[0]

    def __call__(self, x: NDArray[Any], y: NDArray[Any]) -> NDArray[Any]:
        """For x and y points, return the value of the histogram at that point.

        Args:
            x: (*,n) array of x points (can be a meshgrid)
            y: (*,n) array of x points (can be a meshgrid)

        Returns:
            results: (*,n) array of histogram values at the points (x,y).
        """
        # for each input point, find the index of the bin it falls in
        x_index = ((x - self.x_edges[0]) // self.x_bin_width).astype(int)
        y_index = ((y - self.y_edges[0]) // self.y_bin_width).astype(int)

        results = np.zeros_like(x)

        # find indexes in the valid bin range
        valid_x = (x_index >= 0) & (x_index < self.hist.shape[0])
        valid_y = (y_index >= 0) & (y_index < self.hist.shape[1])

        # where both co-ordinates are valid update. Extract only valid results from histogram
        results[valid_x & valid_y] = self.hist[x_index[valid_x], y_index[valid_y]]

        return results


extreme_response_hist = Histogram2DLookup(precalced_erd_x.numpy())

for trace in fig_true_sim.data:
    trace = cast("go.Surface", trace)
    x1 = trace["x"]
    x2 = trace["y"]
    x1_grid, x2_grid = np.meshgrid(x1, x2)
    z = extreme_response_hist(x1_grid, x2_grid)
    _ = trace.update(surfacecolor=z, colorscale="Inferno")

fig_true_sim.show()


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
def get_mean_var(estimator: QoIEstimator, estimates: torch.Tensor | NDArray[Any]) -> tuple[torch.Tensor, torch.Tensor]:
    """TODO: clean this up or delete.

    Args:
        estimator: the QoI function that produced the estimate
        estimates: (*b, n_estimator)

    Returns:
        tensor1: the mean of the estimates, with shape *b
        tensor1: the variance of the estimates, with shape *b

    """
    if not isinstance(estimates, torch.Tensor):
        estimates = torch.tensor(estimates)

    mean = estimator.posterior_sampler.mean(estimates, -1)  # type: ignore[attr-defined] # pyright: ignore[reportAttributeAccessIssue]
    var = estimator.posterior_sampler.var(estimates, -1)  # type: ignore[attr-defined] # pyright: ignore[reportAttributeAccessIssue]

    return mean, var


# %%
_, axes = plt.subplots(nrows=len(n_training_points), sharex=True, figsize=(6, 6 * len(n_training_points)))

for ax, estimate, n_points in zip(axes, results, n_training_points, strict=True):
    mean, var = get_mean_var(qoi_estimator, torch.tensor(estimate))  # type: ignore[assignment]
    qoi_dist = Normal(mean, var**0.5)
    _ = population_estimators.plot_dist(qoi_dist, ax=ax, c="tab:blue", label="QOI estimate")

    ax.axvline(brute_force_qoi_estimate, c="orange", label="Brute force results")

    ax.set_title(f"QoI estimate with {n_points} training points")
    ax.set_xlabel("Response")
    ax.set_ylabel("Density")
    ax.legend()

# Note: if you are interested, the following allows you to see the result of the QoIEstimator if uncertainty is ignored.
# this may not be a very good estimate if the simulator mean doesn't match the surrogate.
# from axtreme.eval.qoi import qoi_ignoring_gp_uncertainty  # noqa: ERA001
# result_no_gp_uncertainty = qoi_ignoring_gp_uncertainty(qoi_estimator_gp_brute_force, model)  # noqa: ERA001

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


def run_trials(
    experiment: Experiment,
    warm_up_generator: Callable[[Experiment], GeneratorRun],
    doe_generator: Callable[[Experiment], GeneratorRun],
    warm_up_runs: int = 3,
    doe_runs: int = 15,
    stopping_criteria: Callable[[Experiment], bool] | None = None,
) -> int:
    """Helper function for running trials for an experiment and returning the QOI results using QoI metric.

    Args:
        experiment: Experiment to perform DOE on.
        warm_up_generator: Generator to create the initial training data on the experiment (e.g., Sobol).
        doe_generator: The generator being used to perform the DoE.
        warm_up_runs: Number of warm-up runs to perform before starting the DoE.
        doe_runs: Number of DoE runs to perform.
        stopping_criteria: Optional function that takes an experiment and returns True if a given
        stopping criteria is met.

    """
    # Warm-up phase
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

        # Check stopping criteria after each DoE iteration
        if stopping_criteria is not None and stopping_criteria(experiment):
            print(f"Stopping criteria met after {i} DoE iterations")
            return i

    return doe_runs + warm_up_runs


def sem_stopping_criteria(experiment: Experiment, sem_threshold: float = 1) -> bool:
    """Stopping criteria based on standard error of the mean (SEM) of QoI metric.

    Args:
        experiment: The experiment to check
        sem_threshold: Threshold for SEM below which to stop

    Returns:
        True if stopping criteria is met (SEM below threshold), False otherwise
    """
    metrics = experiment.fetch_data()
    qoi_metrics = metrics.df[metrics.df["metric_name"] == "QoIMetric"]

    if len(qoi_metrics) == 0:
        print("No QoIMetric data found in the experiment.")
        return False

    # Get the latest QoI metric result
    latest_qoi = qoi_metrics.iloc[-1]
    # DEBUG
    print(f"Latest QoI metric: {latest_qoi['mean']:.4f} with SEM: {latest_qoi['sem']:.4f}")

    if pd.notna(latest_qoi["sem"]) and latest_qoi["sem"] <= sem_threshold:
        print(f"SEM threshold met: {latest_qoi['sem']:.4f} <= {sem_threshold}")
        return True

    return False


# %% [markdown]
# How many iterations to run in the following DOEs

# %%
n_iter = 40
warm_up_runs = 3


# %% [markdown]
# ### Sobol model
# Surrogate trained without and a acquisition function as a comparative baseline.

# %%
# Create QoI tracking metric for tracking of the QoI estimate over the course of the experiment.
QOI_METRIC = QoIMetric(
    name="QoIMetric", qoi_estimator=qoi_estimator, minimum_data_points=warm_up_runs, attach_transforms=True
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


# %% [markdown]
## Run Experiment until a given stopping criteria is met.
# This is optional, but can be useful if you have a specific stopping criteria in mind.


last_itr = run_trials(
    experiment=exp_sobol,
    warm_up_generator=sobol_generator_run,
    doe_generator=sobol_generator_run,
    warm_up_runs=warm_up_runs,
    doe_runs=n_iter,
    stopping_criteria=sem_stopping_criteria,  # Optional: use a stopping criteria based on SEM
)

# %%
# Plot the surface of after warmup and final trial.
surface_warm_up = plot_gp_fits_2d_surface_from_experiment(
    exp_sobol, warm_up_runs, {"loc": _true_loc_func, "scale": _true_scale_func}
)
surface_warm_up.show()
surface_final_trial = plot_gp_fits_2d_surface_from_experiment(
    exp_sobol, last_itr, {"loc": _true_loc_func, "scale": _true_scale_func}
)
surface_final_trial.show()


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
_ = ax.view_init(elev=30, azim=45)  # type: ignore[attr-defined]
_ = ax.plot_surface(grid_x1, grid_x2, scores, cmap="viridis", edgecolor="none")  # type: ignore[attr-defined]
_ = ax.set_xlabel("x1")
_ = ax.set_ylabel("x2")
_ = ax.set_zlabel("score")  # type: ignore[attr-defined]
_ = ax.set_title("Score surface plot")
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


def look_ahead_generator_run(experiment: Experiment) -> GeneratorRun:
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
    qoi_estimator.input_transform = input_transform
    qoi_estimator.outcome_transform = outcome_transform

    # Building the model with the QoILookAhead acquisition function
    model_bridge_cust_ac = Models.BOTORCH_MODULAR(
        experiment=experiment,
        data=experiment.fetch_data(),
        botorch_acqf_class=acqf_class,
        fit_tracking_metrics=False,
        acquisition_options={
            "qoi_estimator": qoi_estimator,
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

last_itr = run_trials(
    experiment=exp_look_ahead,
    warm_up_generator=create_sobol_generator(sobol),
    doe_generator=look_ahead_generator_run,
    warm_up_runs=warm_up_runs,
    doe_runs=n_iter,
    stopping_criteria=sem_stopping_criteria,  # Optional: use a stopping criteria based on SEM
)

# %%
# Plot the surface of after warmup and final trial.
surface_warm_up = plot_gp_fits_2d_surface_from_experiment(
    exp_look_ahead, warm_up_runs, {"loc": _true_loc_func, "scale": _true_scale_func}
)
surface_warm_up.show()
surface_final_trial = plot_gp_fits_2d_surface_from_experiment(
    exp_look_ahead, last_itr, {"loc": _true_loc_func, "scale": _true_scale_func}
)
surface_final_trial.show()

# %% [markdown]
# ### Plot the results


# %%
ax = plot_qoi_estimates_from_experiment(exp_sobol, name="Sobol")
ax = plot_qoi_estimates_from_experiment(exp_look_ahead, ax=ax, color="green", name="look ahead")
_ = ax.axhline(float(brute_force_qoi_estimate), c="black", label="brute_force_value")
_ = ax.set_xlabel("Number of DOE iterations")
_ = ax.set_ylabel("Response")
_ = ax.legend()


# %% [markdown]
### Find the DOE iteration at which the QoI converged to a given percentage of the brute force estimate.
# Here you can see the benefit of using the custom acquisition function. The look-ahead experiment reaches the same
# uncertainty in the QoI estimate with fewer iterations than the Sobol experiment.
# This is because the look-ahead acquisition function selects points that
# are expected to reduce uncertainty in the QoI estimate more effectively than random sampling.


# %%
def find_convergence_trial(experiment: Experiment, uncertainty_threshold_percent: float = 10.0) -> int:
    """Find the trial index when uncertainty falls below threshold percentage of brute force QoI."""
    metrics = experiment.fetch_data()
    qoi_metrics = metrics.df[metrics.df["metric_name"] == "QoIMetric"]

    if len(qoi_metrics) == 0:
        print("No QoIMetric data found in the experiment.")
        return 0

    threshold = abs(brute_force_qoi_estimate) * (uncertainty_threshold_percent / 100.0)

    for _, row in qoi_metrics.iterrows():
        if pd.notna(row["sem"]) and (1.96 * row["sem"]) <= threshold:
            return int(row["trial_index"])
    print(f"No convergence trial found for uncertainty threshold {uncertainty_threshold_percent}% of brute force QoI.")
    return 0


# Find trial where the uncertainty is below a given percentage of the brute force estimate.
uncertainty_threshold_percent = 10.0
sobol_trial = find_convergence_trial(exp_sobol, uncertainty_threshold_percent)
lookahead_trial = find_convergence_trial(exp_look_ahead, uncertainty_threshold_percent)
print(f"Sobol reached uncertainty threshold {uncertainty_threshold_percent}% at trial: {sobol_trial}")
print(f"Look-ahead reached uncertainty threshold {uncertainty_threshold_percent}% at trial: {lookahead_trial}")

# %%
# PLot the surfaces of the Sobol and Look-ahead experiments when both experiments have the same uncertainty in
# the QoI estimate. As one can observe, fewer points are added to the GP in Look-ahead experiment in order to reach
# the same uncertainty in the QoI estimate.
# %%
surface_final_trial = plot_gp_fits_2d_surface_from_experiment(
    exp_sobol, sobol_trial, {"loc": _true_loc_func, "scale": _true_scale_func}
)
surface_final_trial.show()

# %%
surface_final_trial = plot_gp_fits_2d_surface_from_experiment(
    exp_look_ahead, lookahead_trial, {"loc": _true_loc_func, "scale": _true_scale_func}
)
surface_final_trial.show()

# %%
# plot results of the two experiments when convergence is achieved.
ax = plot_qoi_estimates_from_experiment(exp_sobol, name="Sobol", trial_index=sobol_trial)
ax = plot_qoi_estimates_from_experiment(
    exp_look_ahead, ax=ax, color="green", name="look ahead", trial_index=lookahead_trial
)
_ = ax.axhline(float(brute_force_qoi_estimate), c="black", label="brute_force_value")
_ = ax.set_xlabel("Number of DOE iterations")
_ = ax.set_ylabel("Response")
_ = ax.legend()


# %%
## Analysis of Sobol vs LookAhead point selection
# Below is a comparison of the points selected by the Sobol and LookAhead acquisition functions with overlays
# of the environment data and the extreme responses. This allows us to visually assess how well the points cover
# the search space and how they relate to the density of the environment data and extreme responses.


def plot_2dtrials(exp: Experiment, ax: Axes | None = None, colour: str = "blue", label: str | None = None) -> Axes:
    """Plot the points and number the datapoints added over DOE."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    trials = []
    trial_indices = []

    for trial_idx, trial in exp.trials.items():
        if trial.arm:  # type: ignore  # noqa: PGH003
            params = trial.arm.parameters.values()  # type: ignore  # noqa: PGH003
            trials.append(list(params))
            trial_indices.append(trial_idx)

    points = np.array(trials)
    _ = ax.scatter(points[:, 0], points[:, 1], alpha=0.8, s=30, label=label, c=colour)

    for i, (x, y) in enumerate(points):
        _ = ax.annotate(
            str(trial_indices[i]), (x, y), xytext=(2, 2), textcoords="offset points", fontsize=8, color=colour
        )

    return ax


fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# label all the plots
for ax in axes:
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    _ = plot_2dtrials(exp_sobol, colour="blue", ax=ax, label="Sobol")
    _ = plot_2dtrials(exp_look_ahead, colour="forestgreen", ax=ax, label="LookAhead")
    ax.legend()

# Add plot specific info
axes[0].set_title("Points vs Env Density")
axes[0].hist2d(env_data[:, 0], env_data[:, 1], bins=30, alpha=0.6, cmap="Reds", zorder=-1)

axes[1].set_title("Points vs Extreme Response Density")
axes[1].hist2d(precalced_erd_x[:, 0], precalced_erd_x[:, 1], bins=30, alpha=0.6, cmap="Purples", zorder=-1)

axes[2].set_title("Points vs Env and Extreme response density")
axes[2].hist2d(env_data[:, 0], env_data[:, 1], bins=30, alpha=0.6, cmap="Reds", zorder=-1)
axes[2].hist2d(precalced_erd_x[:, 0], precalced_erd_x[:, 1], bins=30, alpha=0.5, cmap="Purples", zorder=-1)

plt.tight_layout()
plt.show()
# %%
