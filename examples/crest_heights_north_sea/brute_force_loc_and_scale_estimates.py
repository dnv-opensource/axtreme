"""Generate and save brute force estimates of location and scale functions."""

# %%
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ax import SearchSpace
from ax.core import ParameterConstraint  # type: ignore[import]
from matplotlib.figure import Figure
from numpy.typing import NDArray
from problem import SEARCH_SPACE  # type: ignore[import]
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import gumbel_r
from simulator import max_crest_height_simulator_function  # type: ignore[import]
from tqdm import tqdm

SAVE_DIR = Path("results/doe")
DEFAULT_FILENAME = "brute_force_loc_scale_data.npz"


# TODO(@henrikstoklandberg 2025-05-12): Add seed to
# #simulator function when implemnted seeded simulator
def generate_and_save_static_dataset(
    search_space: SearchSpace,
    save_dir: Path = SAVE_DIR,
    filename: str = DEFAULT_FILENAME,
    grid_size: int = 1000,
    n_samples: int = 1000,
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

    Saved Dataset Structure:
        The saved .npz dataset contains the following arrays:
        - hs_values: 1D array of Hs (significant wave height) values
        - tp_values: 1D array of Tp (peak wave period) values
        - loc_values: 2D grid of fitted Gumbel location parameters
        - scale_values: 2D grid of fitted Gumbel scale parameters
        - param_names: Names of the parameters as strings
    """
    hs_param = search_space.parameters["Hs"]
    tp_param = search_space.parameters["Tp"]

    hs_range = (hs_param.lower, hs_param.upper)  # type: ignore  # noqa: PGH003
    tp_range = (tp_param.lower, tp_param.upper)  # type: ignore  # noqa: PGH003

    param_names = [hs_param.name, tp_param.name]

    hs_values = np.linspace(hs_range[0], hs_range[1], grid_size)
    tp_values = np.linspace(tp_range[0], tp_range[1], grid_size)
    hs_grid, tp_grid = np.meshgrid(hs_values, tp_values)

    # Extract constraints
    constraints = []
    if hasattr(search_space, "parameter_constraints"):
        constraints = search_space.parameter_constraints

    validity_mask = create_validity_mask(
        constraints=constraints,
        hs_grid=hs_grid,
        tp_grid=tp_grid,
        grid_size=grid_size,
    )

    # Initialize result arrays as full grid but with NaN for invalid points
    loc_values = np.full_like(hs_grid, np.nan, dtype=float)
    scale_values = np.full_like(hs_grid, np.nan, dtype=float)

    # fit Gumbel distribution parameters from simulator for valid grid points
    for row_idx in tqdm(range(grid_size)):
        for col_idx in range(grid_size):
            if not validity_mask[row_idx, col_idx]:
                continue  # Skip invalid points
            hs = hs_grid[row_idx, col_idx]
            tp = tp_grid[row_idx, col_idx]
            x = np.full((n_samples, 2), [hs, tp])
            results = max_crest_height_simulator_function(x)
            loc, scale = gumbel_r.fit(results)
            loc_values[row_idx, col_idx] = loc
            scale_values[row_idx, col_idx] = scale

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


def create_validity_mask(
    constraints: list[ParameterConstraint], hs_grid: NDArray[np.float64], tp_grid: NDArray[np.float64], grid_size: int
) -> NDArray[np.bool_]:
    """Create validity mask with marked invalid points in the grid based on parameter constraints.

    Args:
        constraints: List of ParameterConstraint objects defining the constraints.
        hs_grid: 2D array of Hs values.
        tp_grid: 2D array of Tp values.
        grid_size: Size of the grid (number of points in each dimension).

    Returns:
        2D boolean array where True indicates valid points and False indicates invalid points.
    """
    # Create mask to track valid points
    validity_mask = np.ones((grid_size, grid_size), dtype=bool)

    # Mark invalid points based on constraints
    for row_idx in range(grid_size):
        for col_idx in range(grid_size):
            hs = hs_grid[row_idx, col_idx]
            tp = tp_grid[row_idx, col_idx]

            # Check if point satisfies all constraints
            for constraint in constraints:
                constraint_value = 0
                for param_name, coef in constraint.constraint_dict.items():
                    if param_name == "Hs":
                        constraint_value += coef * hs
                    elif param_name == "Tp":
                        constraint_value += coef * tp

                if constraint_value > constraint.bound:
                    validity_mask[row_idx, col_idx] = False
                    break
    return validity_mask


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

    Expected Dataset Format:
        The .npz file should contain:
        - hs_values: 1D array of Hs grid points
        - tp_values: 1D array of Tp grid points
        - loc_values: 2D array of location parameters at each grid point
        - scale_values: 2D array of scale parameters at each grid point
        - param_names: 1D array of parameter names as strings
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

    Dataset Structure:
        See documentation for generate_and_save_static_dataset() for details on dataset format.
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
    search_space = SEARCH_SPACE

    # Generate and save the static dataset
    dataset_path = generate_and_save_static_dataset(search_space)

    functions = create_functions_from_static_dataset(dataset_path)

    functions = get_brute_force_loc_and_scale_functions(search_space)
    print("functions", functions)

    _ = visualize_brute_force_loc_scale_data()

# %%
