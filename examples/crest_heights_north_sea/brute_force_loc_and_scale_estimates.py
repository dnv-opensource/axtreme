"""Generate and save brute force estimates of location and scale functions."""

# %%
import json
import pickle
from collections.abc import Callable
from pathlib import Path
from typing import Any

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


def generate_brute_force_loc_scale_estimates(
    search_space: SearchSpace,
    grid_size: int = 1000,
    n_samples: int = 1000,
    water_depth: int = 110,
    sample_period: int = 3,
    seed: int = 42,
) -> dict[str, NDArray[np.float64] | Callable[[NDArray[np.float64]], NDArray[np.float64]] | list[str]]:
    """Generate brute force estimates of location and scale functions.

    Args:
        search_space: Ax SearchSpace object defining the parameter ranges
        grid_size: Number of grid points in each dimension
        n_samples: Number of simulator samples per grid point
        water_depth: Water depth in meters
        sample_period: Sample period in hours
        seed: Random seed

    Returns:
        Dictionary with grid data and interpolation functions
    """
    # Set random seed for reproducibility
    rng = np.random.default_rng(seed)

    # Extract parameters from the dictionary
    param_names = list(search_space.parameters.keys())

    # Get the Hs and Tp parameters (assuming they're named "Hs" and "Tp")
    hs_param = search_space.parameters["Hs"]
    tp_param = search_space.parameters["Tp"]

    hs_range = (hs_param.lower, hs_param.upper)  # type: ignore  # noqa: PGH003
    tp_range = (tp_param.lower, tp_param.upper)  # type: ignore  # noqa: PGH003

    param_names = [hs_param.name, tp_param.name]

    # Create grid
    hs_values = np.linspace(hs_range[0], hs_range[1], grid_size)
    tp_values = np.linspace(tp_range[0], tp_range[1], grid_size)
    hs_grid, tp_grid = np.meshgrid(hs_values, tp_values)

    # Initialize arrays to store results
    loc_values = np.zeros_like(hs_grid)
    scale_values = np.zeros_like(hs_grid)

    # Run simulations at each grid point
    print(f"Running {grid_size}x{grid_size} grid with {n_samples} samples per point...")
    for i in tqdm(range(grid_size)):
        for j in range(grid_size):
            hs = hs_grid[i, j]
            tp = tp_grid[i, j]

            # Create input array for simulator (n_samples points of the same sea state)
            x = rng.normal(loc=[hs, tp], scale=0, size=(n_samples, 2))

            # Run simulator
            results = max_crest_height_simulator_function(x, water_depth, sample_period)

            # Fit Gumbel distribution to results
            params = gumbel_r.fit(results)
            # For Gumbel: loc is first parameter, scale is second
            loc, scale = params[0], params[1]

            # Store parameters
            loc_values[i, j] = loc
            scale_values[i, j] = scale

    # Create interpolation functions
    loc_interp = RegularGridInterpolator((hs_values, tp_values), loc_values.T)
    scale_interp = RegularGridInterpolator((hs_values, tp_values), scale_values.T)

    # Create wrapper functions that match the expected signature for plot_gp_fits_2d_surface_from_experiment
    def loc_func(x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return location parameter for given points."""
        return loc_interp(x).astype(np.float64)

    def scale_func(x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return scale parameter for given points."""
        return scale_interp(x).astype(np.float64)

    # Return results
    return {
        "hs_values": hs_values,
        "tp_values": tp_values,
        "loc_values": loc_values,
        "scale_values": scale_values,
        "loc_func": loc_func,
        "scale_func": scale_func,
        "param_names": param_names,
    }


def save_brute_force_loc_scale_data(
    data: dict[str, NDArray[np.float64] | list[str]],
    filename: str = "brute_force_loc_scale_data.pkl",
    save_dir: Path = SAVE_DIR,
) -> None:
    """Save brute force data to file."""
    # Save pickle file with the grid data (not the functions)
    save_data = {
        "hs_values": data["hs_values"],
        "tp_values": data["tp_values"],
        "loc_values": data["loc_values"],
        "scale_values": data["scale_values"],
        "param_names": data["param_names"],
    }
    with Path.open(save_dir / filename, "wb") as f:
        pickle.dump(save_data, f)

    print(f"Data saved to {save_dir / filename}")


def load_brute_force_loc_scale_data(
    load_dir: Path = SAVE_DIR, filename: str = "brute_force_loc_scale_data.pkl"
) -> dict[str, NDArray[np.float64] | Callable[[NDArray[np.float64]], NDArray[np.float64]] | list[str]]:
    """Load brute force location and scale data from a file.

    Args:
        load_dir: Directory where the data file is stored.
        filename: Name of the file to load.

    Returns:
        Dictionary containing grid data, interpolation functions, and parameter names.
    """
    with Path.open(load_dir / filename, "rb") as f:
        data = pickle.load(f)  # noqa: S301

    # Recreate interpolation functions
    loc_interp = RegularGridInterpolator((data["hs_values"], data["tp_values"]), data["loc_values"].T)
    scale_interp = RegularGridInterpolator((data["hs_values"], data["tp_values"]), data["scale_values"].T)

    # Add wrapper functions
    def loc_func(x: NDArray[np.float64]) -> NDArray[np.float64]:
        return loc_interp(x).astype(np.float64)

    def scale_func(x: NDArray[np.float64]) -> NDArray[np.float64]:
        return scale_interp(x).astype(np.float64)

    data["loc_func"] = loc_func
    data["scale_func"] = scale_func

    return data


def visualize_brute_force_loc_scale_data(
    data: dict[str, NDArray[np.float64] | list[str]], save_dir: Path = SAVE_DIR
) -> Figure:
    """Visualize brute force data as surface plots."""
    fig = plt.figure(figsize=(15, 7))

    # Plot location function
    ax1 = fig.add_subplot(121, projection="3d")
    hs_grid, tp_grid = np.meshgrid(data["hs_values"], data["tp_values"])
    surf1 = ax1.plot_surface(hs_grid, tp_grid, data["loc_values"], cmap="viridis")  # type: ignore[attr-defined]
    _ = ax1.set_xlabel("Hs")
    _ = ax1.set_ylabel("Tp")
    _ = ax1.set_zlabel("Location")  # type: ignore[attr-defined]
    _ = ax1.set_title("Location Parameter")
    _ = fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)

    # Plot scale function
    ax2 = fig.add_subplot(122, projection="3d")
    surf2 = ax2.plot_surface(hs_grid, tp_grid, data["scale_values"], cmap="plasma")  # type: ignore[attr-defined]
    _ = ax2.set_xlabel("Hs")
    _ = ax2.set_ylabel("Tp")
    _ = ax2.set_zlabel("Scale")  # type: ignore[attr-defined]
    _ = ax2.set_title("Scale Parameter")
    _ = fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)

    plt.tight_layout()

    # Save plot
    plot_dir = Path(save_dir / "plots")
    plot_dir.mkdir(exist_ok=True)
    plt.savefig(plot_dir / "brute_force_loc_scale_functions.png", dpi=300)

    return fig


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy data types."""

    def default(self, o: Any) -> Any:  # noqa: ANN401
        """Serialize NumPy objects to JSON-compatible formats."""
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def save_brute_force_loc_scale_estimates(search_space: SearchSpace, save_dir: Path = SAVE_DIR) -> dict[str, Any]:
    """Run the brute force estimation process using the current search space."""
    bf_data = generate_brute_force_loc_scale_estimates(
        search_space=search_space,
        grid_size=1000,
        n_samples=1000,
    )
    save_brute_force_loc_scale_data(bf_data, save_dir=save_dir)  # type: ignore[arg-type]

    # Visualize the data
    _ = visualize_brute_force_loc_scale_data(bf_data)  # type: ignore[arg-type]

    # Return the brute force data
    return {
        "loc": bf_data["loc_func"],
        "scale": bf_data["scale_func"],
    }


def get_brute_force_loc_and_scale_functions() -> dict[str, Callable[[NDArray[np.float64]], NDArray[np.float64]]]:
    """Get the brute force location and scale functions."""
    # Load the brute force data
    bf_data = load_brute_force_loc_scale_data()
    # Cast the function types explicitly to help mypy
    loc_func: Callable[[NDArray[np.float64]], NDArray[np.float64]] = bf_data["loc_func"]  # type: ignore[assignment]
    scale_func: Callable[[NDArray[np.float64]], NDArray[np.float64]] = bf_data["scale_func"]  # type: ignore[assignment]

    # Return the location and scale functions
    return {
        "loc": loc_func,
        "scale": scale_func,
    }


# %%
if __name__ == "__main__":
    # Example usage of the brute force estimation function
    from ax import ParameterType, RangeParameter, SearchSpace

    # Define the search space
    search_space = SearchSpace(
        parameters=[
            RangeParameter(name="Hs", parameter_type=ParameterType.FLOAT, lower=7.5, upper=20),
            RangeParameter(name="Tp", parameter_type=ParameterType.FLOAT, lower=7.5, upper=20),
        ]
    )

    # Run the brute force estimation
    _ = save_brute_force_loc_scale_estimates(search_space)


# %%
