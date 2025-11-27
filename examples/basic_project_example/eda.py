"""Visualise the environment data and simulator that define this problem."""

# %%
from pathlib import Path

import numpy as np
from ax import SearchSpace
from ax.core import ParameterType, RangeParameter
from numpy.typing import NDArray
from plotly.subplots import make_subplots
from scipy.stats import gumbel_r
from simulator import _true_loc_func, _true_scale_func

from axtreme.plotting.gp_fit import plot_surface_over_2d_search_space
from axtreme.plotting.histogram3d import histogram_surface3d

# %%
### Visualise the Environment data
_problem_dir = Path(__file__).parent
data: NDArray[np.float64] = np.load(_problem_dir / "data/environment_distribution.npy")

fig = histogram_surface3d(data)
_ = fig.update_layout(title_text="Environment distribution estimate from samples")
_ = fig.update_layout(scene_aspectmode="cube")

fig.show()

# %%
### Visualise the true underlying function and the simulator
# NOTE: this is only possible because we are working with a toy example.

fig = make_subplots(
    rows=1,
    cols=3,
    specs=[[{"type": "surface"}, {"type": "surface"}, {"type": "surface"}]],
    subplot_titles=("location", "scale", "Gumble value (NOTE: taken at a specific quantile)"),
)

# This search space is only to controlling plotting. `experiment` defines the search space that should for optimisation.
plot_search_space = SearchSpace(
    parameters=[
        RangeParameter(name="x1", parameter_type=ParameterType.FLOAT, lower=0, upper=1),
        RangeParameter(name="x2", parameter_type=ParameterType.FLOAT, lower=0, upper=1),
    ]
)

_ = fig.add_trace(plot_surface_over_2d_search_space(plot_search_space, funcs=[_true_loc_func]).data[0], row=1, col=1)
_ = fig.add_trace(plot_surface_over_2d_search_space(plot_search_space, funcs=[_true_scale_func]).data[0], row=1, col=2)


def gumbel_helper(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Helper to plot a specify portion of the response surface.

    This is possible because we know the internal of the simple simulator.
    """
    return gumbel_r.ppf(q=0.75, loc=_true_loc_func(x), scale=_true_scale_func(x))


_ = fig.add_trace(plot_surface_over_2d_search_space(plot_search_space, funcs=[gumbel_helper]).data[0], row=1, col=3)
_ = fig.update_scenes({"xaxis": {"title": "x1"}, "yaxis": {"title": "x2"}, "zaxis": {"title": "response"}})

_ = fig.update_traces(showscale=False)

fig.show()
# %%
