"""Visualise the data that define this problem."""

# %%

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from axtreme.plotting.histogram3d import histogram_surface3d

# %%
### Visualise the Environment data
_problem_dir = Path(__file__).parent
data: NDArray[np.float64] = np.load(_problem_dir / "data/long_term_distribution.npy")

print("data shape: ", data.shape)
print("data min: ", data.min(axis=0))
print("data max: ", data.max(axis=0))
print("data mean: ", data.mean(axis=0))


fig = histogram_surface3d(data)

_ = fig.update_layout(title_text="Long term distribution estimate from samples")
_ = fig.update_layout(scene_aspectmode="cube")

# Create results directory if it doesn't exist
results_dir = _problem_dir / "results"
results_dir.mkdir(exist_ok=True)

# Save figure as HTML
output_path_html = results_dir / "long_term_distribution_histogram.html"
fig.write_html(str(output_path_html))
print(f"HTML file saved to: {output_path_html}")


fig.show()

# %%
