"""Evaluate the convergence of the QOI for different training datasets."""

# %%
import json
import multiprocessing
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from ax.modelbridge.registry import Models
from numpy.typing import NDArray
from plotly.subplots import make_subplots
from problem import (  # type: ignore[import-not-found]
    brut_force_qoi,
    dataset,
    hs_bounds,
    make_exp,
    period_length,
    tp_bounds,
)
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import Tensor
from torch.distributions import Normal
from torch.utils.data import DataLoader, Dataset, RandomSampler

from axtreme.data import ImportanceAddedWrapper
from axtreme.data.dataset import MinimalDataset
from axtreme.experiment import add_sobol_points_to_experiment
from axtreme.plotting.histogram3d import histogram_surface3d
from axtreme.qoi import MarginalCDFExtrapolation
from axtreme.qoi.qoi_estimator import QoIEstimator
from axtreme.sampling.importance_sampling import importance_sampling_distribution_uniform_region
from axtreme.sampling.ut_sampler import UTSampler
from axtreme.utils import population_estimators, transforms


# %%
def run_qoi_estimation(dataloader: DataLoader) -> tuple[list[Tensor], QoIEstimator]:
    """Performs QoI estimation for a given dataloader."""
    qoi_estimator = MarginalCDFExtrapolation(
        env_iterable=dataloader,
        period_len=period_length,
        quantile=torch.tensor(quantile),
        quantile_accuracy=torch.tensor(0.01),
        posterior_sampler=posterior_sampler,
    )

    results = []

    for points in tqdm.tqdm(n_training_points):
        print(points)
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

        results.append(qoi_estimator(model))
    return results, qoi_estimator


def set_up_env_data_distribution_from_samples(
    env_data: Dataset[NDArray[np.float64]], num_training_samples: int = 10_000
) -> Pipeline:
    """Calculate the probability density function of the environment distribution.

    Args:
        env_data: data for which the distribution shall be estimated
        num_training_samples: number of samples from env_data to be used for estimation,
        needs to be relatively large for a good estimate

    Returns:
        The best estimator for the distribution.
    """
    pipeline = Pipeline(
        [
            ("normalize", StandardScaler()),
            ("regressor", KernelDensity()),
        ]
    )
    params = {"regressor__bandwidth": np.logspace(-1, 1, 20)}

    # Fit the distribution
    grid = GridSearchCV(pipeline, params, n_jobs=-1)

    # Only use a subset of the data to save time
    samples = torch.tensor(env_data)
    training_samples = samples[torch.randint(0, samples.shape[0], (num_training_samples,))]

    _ = grid.fit(training_samples)

    return grid.best_estimator_


def env_distribution_pdf(x: torch.Tensor, kde_pipeline: Pipeline) -> torch.Tensor:
    """Calculate the probability density function of the environment distribution."""
    return torch.exp(torch.tensor(kde_pipeline.score_samples(x)))


def generate_importance_samples(
    env_data: Dataset[NDArray[np.float64]],
    hs_bounds: tuple[float, float],
    tp_bounds: tuple[float, float],
    threshold: float = 1e-10,
    num_importance_samples: int = int(1e5),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate importance samples using the environment distribution PDF."""
    kde_pipeline = set_up_env_data_distribution_from_samples(env_data)

    region = torch.tensor([[float(hs_bounds[0]), float(tp_bounds[0])], [float(hs_bounds[1]), float(tp_bounds[1])]])

    importance_samples, importance_weights = importance_sampling_distribution_uniform_region(
        lambda x: env_distribution_pdf(x, kde_pipeline), region, threshold, num_importance_samples
    )
    return importance_samples, importance_weights


def plot_importance_vs_env_data(dataset: Dataset[NDArray[np.float64]], importance_samples: Tensor) -> None:
    """Plot importance sample dataset vs whole env data."""
    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "surface"}, {"type": "surface"}]],
        subplot_titles=["env_data Distribution", "Importance Sampling Distribution"],
    )

    # Add surface for env_data
    surf1 = histogram_surface3d(dataset.data, n_bins=20).data[0]  # type: ignore[attr-defined]
    _ = surf1.update(coloraxis="coloraxis1")
    _ = fig.add_trace(surf1, row=1, col=1)

    # Add surface for importance_samples
    surf2 = histogram_surface3d(importance_samples.numpy(), n_bins=20).data[0]
    _ = surf2.update(coloraxis="coloraxis2")
    _ = fig.add_trace(surf2, row=1, col=2)

    _ = fig.update_layout(
        scene={"aspectmode": "cube"},
        scene2={"aspectmode": "cube"},
        coloraxis1={
            "colorscale": "Viridis",
            "colorbar": {"title": "PDF", "x": 0.40, "xanchor": "left"},
        },
        coloraxis2={
            "colorscale": "Viridis",
            "colorbar": {"title": "PDF", "x": 1, "xanchor": "left"},
        },
    )

    fig.show()


def plot_importance_sampling_vs_important_region(importance_samples: Tensor) -> None:
    """Plot Importance sample vs location of extreme response."""
    # Show that the importance sampling is covering the important region of max response
    brut_force_file_path = Path(__file__).parent / f"usecase/results/brute_force/{period_length}_period_length.json"
    with brut_force_file_path.open() as fp:
        brute_force_results = json.load(fp)
    max_location = torch.tensor(brute_force_results["env_data"])

    _ = plt.scatter(
        max_location[:, 0], max_location[:, 1], s=5, color="red", alpha=0.5, label="extrem response location"
    )
    _ = plt.scatter(
        importance_samples[:, 0], importance_samples[:, 1], s=5, color="grey", alpha=0.2, label="importance sample"
    )

    _ = plt.title("Extrem value location")  # type: ignore[assignment]
    _ = plt.xlabel("Hs")  # type: ignore[assignment]
    _ = plt.ylabel("Tp")  # type: ignore[assignment]

    _ = plt.legend()
    plt.savefig("usecase/results/importance_samples/importance_sampling_extrem_response_coverage.png")


# Compare QoI estimation with and without importance sampling
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


def plot_qoi_distribution(  # noqa: PLR0913
    n_training_points: list[int],
    combined_qoi_estimator_importance_sample: list,
    combined_results_importance_sample: list,
    combined_qoi_estimator_whole_dataset: list,
    combined_results_whole_dataset: list,
    brut_force_qoi: float,
    num_samples: int,
    num_iterations: int,
) -> None:
    """Plots the QoI distribution estimates with and without importance sampling."""
    fig, axes = plt.subplots(nrows=len(n_training_points), sharex=True, figsize=(6, 6 * len(n_training_points)))

    for idx, (ax, n_points) in enumerate(zip(axes, n_training_points, strict=True)):
        for iter_idx in range(num_iterations):
            # Importance sampling distribution
            mean_importance, var_importance = get_mean_var(
                combined_qoi_estimator_importance_sample[iter_idx],
                torch.tensor(combined_results_importance_sample[iter_idx][idx]),
            )
            qoi_dist_importance = Normal(mean_importance, var_importance**0.5)
            _ = population_estimators.plot_dist(
                qoi_dist_importance,
                ax=ax,
                c="tab:red",
                label="QOI estimate (importance sampling)" if iter_idx == 0 else None,
            )

            # Whole dataset distribution
            mean_whole, var_whole = get_mean_var(
                combined_qoi_estimator_whole_dataset[iter_idx],
                torch.tensor(combined_results_whole_dataset[iter_idx][idx]),
            )
            qoi_dist_whole = Normal(mean_whole, var_whole**0.5)
            _ = population_estimators.plot_dist(
                qoi_dist_whole,
                ax=ax,
                c="tab:blue",
                label="QOI estimate" if iter_idx == 0 else None,
            )

        ax.axvline(brut_force_qoi, c="orange", label="Brute force results")
        ax.set_title(f"QoI estimate with {n_points} training points")
        ax.set_xlabel("Response")
        ax.set_ylabel("Density")
        ax.legend()

    fig.savefig(f"usecase/results/qoi/combined_qoi_distribution_{num_samples}_num_samples.png")


def plot_qoi_histogram(
    n_training_points: list[int],
    combined_results_importance_sample: list,
    combined_results_whole_dataset: list,
    brut_force_qoi: float,
    num_iterations: int,
) -> None:
    """Plots the histogram of QoI estimates with and without importance sampling."""
    fig, axes = plt.subplots(nrows=len(n_training_points), sharex=True, figsize=(6, 6 * len(n_training_points)))

    for idx, (ax, n_points) in enumerate(zip(axes, n_training_points, strict=True)):
        importance_sample_mean = [
            combined_results_importance_sample[iter_idx][idx].mean() for iter_idx in range(num_iterations)
        ]
        whole_dataset_mean = [
            combined_results_whole_dataset[iter_idx][idx].mean() for iter_idx in range(num_iterations)
        ]
        ax.hist(importance_sample_mean, color="red", bins=10, label="QOI estimate with importance sampling", alpha=0.5)
        ax.hist(whole_dataset_mean, color="blue", bins=10, label="QOI estimate", alpha=0.5)
        ax.axvline(brut_force_qoi, c="orange", label="Brute force results")
        ax.set_title(f"QoI estimate with {n_points} training points")
        ax.set_xlabel("Response")
        ax.set_ylabel("Density")
        ax.legend()

    fig.savefig(f"usecase/results/qoi/combined_qoi_histogram_{num_samples}_num_samples.png")


def estimate_qoi(
    dataset: Dataset[NDArray[np.float64]],
    importance_dataset: Dataset[NDArray[np.float64]],
    num_samples: int,
    batch_size: int,
) -> tuple[list[Tensor], QoIEstimator, list[Tensor], QoIEstimator]:
    """Perform QoI estimation with and without importance sampling.

    Args:
        dataset: whole env dataset
        importance_dataset: dataset containing both the importance samples and weights
        num_samples: number of samples to draw in Dataloader
        batch_size: batch size for Dataloader

    Returns:
        QoI results with and with and without importance sampling and the corresponding QoI estimators

    """
    # Step 1: Run QoI without importance sampling
    sampler_whole_dataset = RandomSampler(dataset, num_samples=num_samples, replacement=True)  # type: ignore[arg-type]
    dataloader_whole_dataset = DataLoader(dataset, sampler=sampler_whole_dataset, batch_size=batch_size)

    results_whole_dataset, qoi_estimator_whole_dataset = run_qoi_estimation(dataloader_whole_dataset)

    # Step 2: Run QoI with importance samples
    importance_sample_sampler = RandomSampler(importance_dataset, num_samples=num_samples, replacement=True)  # type: ignore[arg-type]
    importance_sample_dataloader = DataLoader(
        importance_dataset, sampler=importance_sample_sampler, batch_size=batch_size
    )

    results_importance_sample, qoi_estimator_importance_sample = run_qoi_estimation(importance_sample_dataloader)

    return (
        results_whole_dataset,
        qoi_estimator_whole_dataset,
        results_importance_sample,
        qoi_estimator_importance_sample,
    )


def run_parallel_qoi_estimation(
    importance_dataset: Dataset[NDArray[np.float64]],
    dataset: Dataset[NDArray[np.float64]],
    num_samples: int,
    batch_size: int,
    num_iterations: int,
    num_process: int | None = None,
) -> tuple[list[list[Tensor]], list[QoIEstimator], list[list[Tensor]], list[QoIEstimator]]:
    """Run QoI estimation with and without importance sampling in parallel for num_iterations iterations.

    Args:
        importance_dataset: dataset containing both the importance samples and weights
        dataset: whole env dataset
        num_samples: number of samples to draw in Dataloader
        batch_size: batch size for Dataloader
        num_iterations: How often the QoI estimation shall be run
        num_process: How many processes shall be run in parallel. As a default all available ones minus one are used

    Returns:
        QoI results with and with and without importance sampling and the corresponding QoI estimators
        for num_iterations iterations

    """
    if num_process is None:
        num_process = multiprocessing.cpu_count() - 1

    combined_results_whole = []
    combined_estimators_whole = []
    combined_results_importance = []
    combined_estimators_importance = []

    with multiprocessing.Pool(processes=num_process) as pool:
        results = pool.starmap(
            estimate_qoi, [(dataset, importance_dataset, num_samples, batch_size) for _ in range(num_iterations)]
        )

    for res_whole, est_whole, res_imp, est_imp in results:
        combined_results_whole.append(res_whole)
        combined_estimators_whole.append(est_whole)
        combined_results_importance.append(res_imp)
        combined_estimators_importance.append(est_imp)

    return (
        combined_results_whole,
        combined_estimators_whole,
        combined_results_importance,
        combined_estimators_importance,
    )


# %%
if __name__ == "__main__":
    num_samples = 1_000  # number of samples to be used for sampler
    seed_random_sampler = 10  # Seed used in random sampler
    batch_size = 256  # Batch size for random sampler

    posterior_sampler = UTSampler()

    quantile = np.exp(-1)  # Quantile for QoI

    n_training_points = [50]  # List of number of training points for GP

    # Specify if new importance shall be created or existing be loaded from a file
    new_importance_samples = False

    if new_importance_samples:
        importance_samples, importance_weights = generate_importance_samples(
            dataset,
            hs_bounds,
            tp_bounds,
            threshold=1e-10,
            num_importance_samples=int(1e5),
        )
        np.save(
            f"usecase/results/importance_samples/importance_samples_{num_samples}_num_samples.npy", importance_samples
        )
        np.save(
            f"usecase/results/importance_samples/importance_weights_{num_samples}_num_samples.npy", importance_weights
        )
    else:
        importance_samples = torch.from_numpy(
            np.load(f"usecase/results/importance_samples/importance_samples_{num_samples}_num_samples.npy")
        )
        importance_weights = torch.from_numpy(
            np.load(f"usecase/results/importance_samples/importance_weights_{num_samples}_num_samples.npy")
        )
    # Combine importance samples and weights in one dataset
    importance_dataset = ImportanceAddedWrapper(MinimalDataset(importance_samples), MinimalDataset(importance_weights))

    plot_importance_sampling_vs_important_region(importance_samples)

    # Calculate QoI num_iterations times in parallel with and without importance sampling
    # If all available processors shall we used it makes sense to specify the num_iterations as multiple of that
    num_iterations = 1 * 2

    # Make either a new run or load saved results
    # TODO(@am-kaiser): As of now only the results are saved but not any information about which parameters were used.
    # This makes it hard to know what kind of experiments were run. One option is to define classes as in
    # TDR_rax/examples/demo2d/qoi_estimator_bias_var.py. As an alternative a json file could be created.
    # It would also be nice to have access to the QoIEstimator object for saved results (09.05.25)
    run_qoi = True
    if run_qoi:
        (
            combined_results_whole_dataset,
            combined_qoi_estimator_whole_dataset,
            combined_results_importance_sample,
            combined_qoi_estimator_importance_sample,
        ) = run_parallel_qoi_estimation(importance_dataset, dataset, num_samples, batch_size, num_iterations)

        np.save(
            f"usecase/results/qoi/combined_importance_qoi_results_{num_samples}_num_samples.npy",
            combined_results_importance_sample,
        )
        np.save(
            f"usecase/results/qoi/combined_whole_qoi_results_{num_samples}_num_samples.npy",
            combined_results_whole_dataset,
        )
    else:
        combined_results_importance_sample = np.load(
            f"usecase/results/qoi/combined_importance_qoi_results_{num_samples}_num_samples.npy"
        )
        combined_results_whole_dataset = np.load(
            f"usecase/results/qoi/combined_whole_qoi_results_{num_samples}_num_samples.npy"
        )

    plot_qoi_distribution(
        n_training_points,
        combined_qoi_estimator_importance_sample,
        combined_results_importance_sample,
        combined_qoi_estimator_whole_dataset,
        combined_results_whole_dataset,
        brut_force_qoi,
        num_samples,
        num_iterations,
    )

    plot_qoi_histogram(
        n_training_points,
        combined_results_importance_sample,
        combined_results_whole_dataset,
        brut_force_qoi,
        num_iterations,
    )
# %%
