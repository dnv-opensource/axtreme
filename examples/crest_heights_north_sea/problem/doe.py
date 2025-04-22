# %%  # noqa: D100
from collections.abc import Callable
from typing import Any

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
from botorch.optim import optimize_acqf
from env_data import collect_data  # type: ignore[import-not-found]
from matplotlib.axes import Axes
from numpy.typing import NDArray
from problem import DIST, N_ENV_SAMPLES_PER_PERIOD, SEARCH_SPACE, make_exp  # type: ignore[import-not-found]
from simulator import max_crest_height_simulator_function  # type: ignore[import-not-found]
from torch.utils.data import DataLoader

from axtreme import sampling
from axtreme.acquisition import QoILookAhead
from axtreme.data import FixedRandomSampler, MinimalDataset
from axtreme.experiment import add_sobol_points_to_experiment
from axtreme.plotting.gp_fit import plot_gp_fits_2d_surface
from axtreme.qoi import MarginalCDFExtrapolation
from axtreme.qoi.qoi_estimator import QoIEstimator
from axtreme.sampling.ut_sampler import UTSampler
from axtreme.utils import transforms

torch.set_default_dtype(torch.float64)
device = "cpu"

# %%
search_space = SEARCH_SPACE
n_env_samples_per_period = N_ENV_SAMPLES_PER_PERIOD
dist = DIST


# %%
raw_data: pd.DataFrame = collect_data()

# we convert this data to Numpy for ease of use from here on out
env_data: NDArray[np.float64] = raw_data.to_numpy()

# %%
# Bruteforce estimate placeholder
n_erd_samples = 1000
erd_samples = []
for _ in range(n_erd_samples):
    indices = np.random.choice(env_data.shape[0], size=N_ENV_SAMPLES_PER_PERIOD, replace=True)  # noqa: NPY002
    period_sample = env_data[indices]

    responses = max_crest_height_simulator_function(period_sample)
    erd_samples.append(responses.max())

brute_force_qoi_estimate = np.median(erd_samples)
print(f"Brute force estimate of our QOI is {brute_force_qoi_estimate}")


# %%
# Currently copied from basic_example.py
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
                plot_gp_fits_2d_surface(model_bridge, search_space, {"sim": max_crest_height_simulator_function})
            )
        print(f"iter {i} done")

    for fig in figs:
        fig.show()
    return np.vstack(qoi_results)


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


# %% [markdown]
# How many iterations to run in the following DOEs

# %%
n_iter = 15  # 40
qoi_iter = 1
warm_up_runs = 3

# %% [markdown]
# ### Sobol model
# Surrogate trained without and a acquisition function as a comparative baseline.

# %%
exp_sobol = make_exp()
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

qoi_results_sobol = run_trials(
    experiment=exp_sobol,
    warm_up_generator=sobol_generator_run,
    doe_generator=sobol_generator_run,
    qoi_estimator=qoi_estimator,
    warm_up_runs=warm_up_runs,
    doe_runs=n_iter,
    qoi_iter=qoi_iter,
)


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
scores = acqusition(torch.tensor([[[15.0, 15.0]]]))

# %% Perform the grid search and plot
point_per_dim = 21
Hs = torch.linspace(7.5, 20, point_per_dim)
Tp = torch.linspace(7.5, 20, point_per_dim)
grid_hs, grid_tp = torch.meshgrid(Hs, Tp, indexing="xy")
grid = torch.stack([grid_hs, grid_tp], dim=-1)
# make turn into a shape that can be processsed by the acquisition function
x_candidates = grid.reshape(-1, 1, 2)
acqusition = QoILookAhead(model, qoi_estimator)
scores = acqusition(x_candidates)
scores = scores.reshape(grid.shape[:-1])

# %%
# TODO(@henrikstoklandberg): This is plot looks a bit suprising, should be investigated before the we do the final DOE
# steps.
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")
_ = ax.view_init(elev=30, azim=45)  # pyright: ignore[reportAttributeAccessIssue]
_ = ax.plot_surface(grid_hs, grid_tp, scores, cmap="viridis", edgecolor="none")  # pyright: ignore[reportAttributeAccessIssue]
_ = ax.set_xlabel("x1")
_ = ax.set_ylabel("x2")
_ = ax.set_zlabel("score")  # pyright: ignore[reportAttributeAccessIssue]
_ = ax.set_title("Score surface plot")
print("max_score ", scores.max())


# %% perform a round of optimisation using the under the hood optimiser
candidate, result = optimize_acqf(
    acqusition,
    bounds=torch.tensor([[7.5, 7.5], [20.0, 20.0]]),
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
# This needs to be instantiated outside of the loop so the internal state of the generator persists.
sobol = Models.SOBOL(search_space=exp_look_ahead.search_space, seed=5)

qoi_results_look_ahead = run_trials(
    experiment=exp_look_ahead,
    warm_up_generator=create_sobol_generator(sobol),
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
baseline_mean, baseline_var = get_mean_var(qoi_estimator, qoi_results_sobol)
ax = plot_raw_ut_estimates(baseline_mean, baseline_var, name="Sobol")
lookahead_mean, lookahead_var = get_mean_var(qoi_estimator, qoi_results_look_ahead)
ax = plot_raw_ut_estimates(lookahead_mean, lookahead_var, ax=ax, color="green", name="look ahead")
_ = ax.axhline(brute_force_qoi_estimate, c="black", label="brute_force_value")
_ = ax.set_xlabel("Number of DOE iterations")
_ = ax.set_ylabel("Response")
_ = ax.legend()

# %%
