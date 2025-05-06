"""Evaluate the convergence of the QOI for different training datasets."""

# %%
import matplotlib.pyplot as plt
import torch
from ax import (
    Experiment,
    SearchSpace,
)
from ax.core import ParameterType, RangeParameter
from ax.core.parameter_constraint import ParameterConstraint
from ax.modelbridge.registry import Models
from problem import (  # type: ignore[import-not-found]
    DIST,
    brut_force_qoi,
    dataset,
    period_length,
    sim,
)
from torch.distributions import Normal
from torch.utils.data import DataLoader

from axtreme.data import FixedRandomSampler
from axtreme.experiment import add_sobol_points_to_experiment, make_experiment
from axtreme.qoi import MarginalCDFExtrapolation
from axtreme.qoi.qoi_estimator import QoIEstimator
from axtreme.sampling.ut_sampler import UTSampler
from axtreme.utils import population_estimators, transforms


# %%
def get_mean_var(estimator: QoIEstimator, estimates: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Get mean and variance for an estimator.

    Args:
        estimator: the QoI function that produced the estimate
        estimates: (*b, n_estimator)

    Returns:
        tensor1: the mean of the estimates, with shape *b
        tensor1: the mean of the estimates, with shape *b

    """
    if not isinstance(estimates, torch.Tensor):  # pyright: ignore[reportUnnecessaryIsInstance]
        estimates = torch.tensor(estimates)

    mean = estimator.posterior_sampler.mean(estimates, -1)  # type: ignore[attr-defined]
    var = estimator.posterior_sampler.var(estimates, -1)  # type: ignore[attr-defined]

    return mean, var


## Pick the search space over which to create a surrogate
SEARCH_SPACE = SearchSpace(
    parameters=[
        RangeParameter(name="Hs", parameter_type=ParameterType.FLOAT, lower=0, upper=17),
        RangeParameter(name="Tp", parameter_type=ParameterType.FLOAT, lower=1, upper=32),
    ],
    parameter_constraints=[
        # Linear constraint: Hs <= 1.5 Tp
        ParameterConstraint(constraint_dict={"Hs": 1, "Tp": -1.5}, bound=0.0),
    ],
)


def make_exp() -> Experiment:
    """Convenience function returns a fresh Experiment of this problem."""
    return make_experiment(sim, SEARCH_SPACE, DIST, n_simulations_per_point=1000)


# %%
env_sample_size = [1_000, 5_000, 10_000, 50_000]

for num_samples in env_sample_size:
    sampler = FixedRandomSampler(dataset, num_samples=num_samples, replacement=True)  # type: ignore[arg-type]
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=256)

    posterior_sampler = UTSampler()

    qoi_estimator = MarginalCDFExtrapolation(
        env_iterable=dataloader,
        period_len=period_length,
        quantile=torch.tensor(0.5),
        quantile_accuracy=torch.tensor(0.01),
        posterior_sampler=posterior_sampler,
    )

    n_training_points = [30, 50, 128, 512]
    results = []

    for points in n_training_points:
        exp = make_exp()
        add_sobol_points_to_experiment(exp, n_iter=points, seed=5)
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

    fig, axes = plt.subplots(nrows=len(n_training_points), figsize=(6, 6 * len(n_training_points)))  # , sharex=True

    for ax, estimate, n_points in zip(axes, results, n_training_points, strict=True):
        mean, var = get_mean_var(qoi_estimator, torch.tensor(estimate))
        qoi_dist = Normal(mean, var**0.5)
        _ = population_estimators.plot_dist(qoi_dist, ax=ax, c="tab:blue", label="QOI estimate")

        ax.axvline(brut_force_qoi, c="orange", label="Brute force results")

        ax.set_title(f"QoI estimate with {n_points} training points. num_samples = {num_samples}")
        ax.set_xlabel("Response")
        ax.set_ylabel("Density")
        ax.legend()
    fig.savefig(f"usecase/results/qoi/qoi_estimate_{num_samples}_num_samples.png")

# %%
