"""Generate and save brute force estimates of location and scale functions."""

# %%
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ax import SearchSpace
from matplotlib.figure import Figure
from numpy.typing import NDArray
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import gumbel_r
from simulator import max_crest_height_simulator_function  # type: ignore[import]
from tqdm import tqdm

SAVE_DIR = Path("usecase/data")
PLOT_DIR = Path("results/doe")
DEFAULT_FILENAME = "brute_force_loc_scale_data.npz"


def generate_and_save_static_dataset(
    search_space: SearchSpace,
    save_dir: Path = SAVE_DIR,
    filename: str = DEFAULT_FILENAME,
    grid_size: int = 1000,
    n_samples: int = 1000,
    seed: int = 42,
) -> Path:
    """Generate and save static dataset of location and scale values on a grid.

    Args:
        search_space: Ax SearchSpace object defining the parameter ranges.
        save_dir: Directory to save the dataset.
        filename: Name of the file to save the dataset.
        grid_size: Number of grid points in each dimension.
        n_samples: Number of simulator samples per grid point.
        seed: Random seed.

    Returns:
        Path to the saved dataset file.
    """
    rng = np.random.default_rng(seed)

    hs_param = search_space.parameters["Hs"]
    tp_param = search_space.parameters["Tp"]

    hs_range = (hs_param.lower, hs_param.upper)  # type: ignore  # noqa: PGH003
    tp_range = (tp_param.lower, tp_param.upper)  # type: ignore  # noqa: PGH003

    param_names = [hs_param.name, tp_param.name]

    hs_values = np.linspace(hs_range[0], hs_range[1], grid_size)
    tp_values = np.linspace(tp_range[0], tp_range[1], grid_size)
    hs_grid, tp_grid = np.meshgrid(hs_values, tp_values)

    loc_values = np.zeros_like(hs_grid)
    scale_values = np.zeros_like(hs_grid)

    print(f"Running {grid_size}x{grid_size} grid with {n_samples} samples per point...")
    for i in tqdm(range(grid_size)):
        for j in range(grid_size):
            hs = hs_grid[i, j]
            tp = tp_grid[i, j]
            x = rng.normal(loc=[hs, tp], scale=0, size=(n_samples, 2))
            results = max_crest_height_simulator_function(x)
            loc, scale = gumbel_r.fit(results)[:2]
            loc_values[i, j] = loc
            scale_values[i, j] = scale

    # Save the dataset
    save_data = {
        "hs_values": hs_values,
        "tp_values": tp_values,
        "loc_values": loc_values,
        "scale_values": scale_values,
        "param_names": np.array(param_names, dtype=str),
    }

    save_path = save_dir / filename
    save_dir.mkdir(exist_ok=True, parents=True)
    np.savez_compressed(save_path, **save_data)
    print(f"Data saved to {save_path}")

    return save_path


def create_functions_from_static_dataset(
    dataset_path: Path | None = None,
    load_dir: Path = SAVE_DIR,
    filename: str = DEFAULT_FILENAME,
) -> dict[str, Callable[[NDArray[np.float64]], NDArray[np.float64]]]:
    """Create interpolation functions from a static dataset.

    Args:
        dataset_path: Direct path to the dataset file. If None, constructed from load_dir and filename.
        load_dir: Directory containing the dataset file.
        filename: Name of the dataset file.

    Returns:
        Dictionary with location and scale interpolation functions.
    """
    if dataset_path is None:
        dataset_path = load_dir / filename

    with np.load(dataset_path) as data:
        hs_values = data["hs_values"]
        tp_values = data["tp_values"]
        loc_values = data["loc_values"]
        scale_values = data["scale_values"]

    loc_interp = RegularGridInterpolator((hs_values, tp_values), loc_values.T)
    scale_interp = RegularGridInterpolator((hs_values, tp_values), scale_values.T)

    def loc_func(x: NDArray[np.float64]) -> NDArray[np.float64]:
        return loc_interp(x).astype(np.float64)

    def scale_func(x: NDArray[np.float64]) -> NDArray[np.float64]:
        return scale_interp(x).astype(np.float64)

    return {"loc": loc_func, "scale": scale_func}


def get_brute_force_loc_and_scale_functions(
    search_space: SearchSpace,
    save_dir: Path = SAVE_DIR,
    filename: str = DEFAULT_FILENAME,
    grid_size: int = 1000,
    n_samples: int = 1000,
) -> dict[str, Callable[[NDArray[np.float64]], NDArray[np.float64]]]:
    """Get location and scale functions, generating the dataset if it doesn't exist.

    Args:
        search_space: Ax SearchSpace object defining the parameter ranges. Required if dataset doesn't exist.
        save_dir: Directory for saving/loading the dataset.
        filename: Name of the dataset file.
        grid_size: Number of grid points if generating a new dataset.
        n_samples: Number of simulator samples per grid point if generating a new dataset.

    Returns:
        Dictionary with location and scale interpolation functions.
    """
    dataset_path = save_dir / filename

    # If dataset doesn't exist, generate it
    if not dataset_path.exists():
        dataset_path = generate_and_save_static_dataset(
            search_space=search_space,
            save_dir=save_dir,
            filename=filename,
            grid_size=grid_size,
            n_samples=n_samples,
        )

    # Create and return the functions
    return create_functions_from_static_dataset(dataset_path)


def visualize_brute_force_loc_scale_data(
    dataset_path: Path | None = None,
    load_dir: Path = SAVE_DIR,
    filename: str = DEFAULT_FILENAME,
) -> Figure:
    """Visualize brute force data as surface plots.

    Args:
        dataset_path: Direct path to the dataset file. If None, constructed from load_dir and filename.
        load_dir: Directory containing the dataset file.
        filename: Name of the dataset file.
        save_dir: Directory to save the plots.

    Returns:
        Figure object of the plot.
    """
    # Resolve dataset path
    if dataset_path is None:
        dataset_path = load_dir / filename

    # Load data from npz file
    with np.load(dataset_path) as data:
        hs_values = data["hs_values"]
        tp_values = data["tp_values"]
        loc_values = data["loc_values"]
        scale_values = data["scale_values"]

    # Create figure
    fig = plt.figure(figsize=(15, 7))

    # Plot location function
    ax1 = fig.add_subplot(121, projection="3d")
    hs_grid, tp_grid = np.meshgrid(hs_values, tp_values)
    surf1 = ax1.plot_surface(hs_grid, tp_grid, loc_values, cmap="viridis")  # type: ignore[attr-defined]
    _ = ax1.set_xlabel("Hs")
    _ = ax1.set_ylabel("Tp")
    _ = ax1.set_zlabel("Location")  # type: ignore[attr-defined]
    _ = ax1.set_title("Location Parameter")
    _ = fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)

    # Plot scale function
    ax2 = fig.add_subplot(122, projection="3d")
    surf2 = ax2.plot_surface(hs_grid, tp_grid, scale_values, cmap="plasma")  # type: ignore[attr-defined]
    _ = ax2.set_xlabel("Hs")
    _ = ax2.set_ylabel("Tp")
    _ = ax2.set_zlabel("Scale")  # type: ignore[attr-defined]
    _ = ax2.set_title("Scale Parameter")
    _ = fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)

    plt.tight_layout()

    plot_dir = Path(load_dir / "plots")
    plot_dir.mkdir(exist_ok=True)
    plt.savefig(plot_dir / "brute_force_loc_scale_functions.png", dpi=300)
    print(f"Plot saved to {plot_dir / 'brute_force_loc_scale_functions.png'}")

    return fig


# %% Main
if __name__ == "__main__":
    from ax import ParameterType, RangeParameter, SearchSpace

    search_space = SearchSpace(
        parameters=[
            RangeParameter(name="Hs", parameter_type=ParameterType.FLOAT, lower=7.5, upper=20),
            RangeParameter(name="Tp", parameter_type=ParameterType.FLOAT, lower=7.5, upper=20),
        ]
    )

    # Generate and save the static dataset
    dataset_path = generate_and_save_static_dataset(search_space)

    functions = create_functions_from_static_dataset(dataset_path)

    functions = get_brute_force_loc_and_scale_functions(search_space)
    print("functions", functions)

    _ = visualize_brute_force_loc_scale_data()
