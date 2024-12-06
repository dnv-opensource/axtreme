"""Helper function for plotting DoE behaviour."""

from typing import Any, TypeAlias

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

Numpy2dArray: TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.float64]]


def plot_qoi_estimates(
    results: Numpy2dArray,
    ax: None | Axes = None,
    q: tuple[float, ...] = (0.1, 0.5, 0.9),
    points_between_ests: int = 1,
    name: str | None = None,
    **kwargs: Any,  # noqa: ANN401
) -> Axes:
    """Plots how the QoI estimates changes over DoE process.

    Args:
        results: shape (n_doe_rounds, n_qoi_estimates).

            - n_doe_rounds: The number of DoE rounds in which a QoI estimate was produced.
            - n_qoi_estimates: the number of estimates produced by a single run of the QoI estimator.

        ax: ax to add the plots to. If not provided, one will be created.
        q: the quantiles that should be used/reported.
        points_between_ests: This should be used if multplie DoE iterations are used between qoi estimates
            (e.g if the estimate is expensive). It adjusts the scale of the x axis.
        name: optional name that should be added to the legend information for this plot
        kwargs: kwargs that should be passed to matplotlib. Must be applicable to `ax.plot` and `ax.fill_between`

    Returns:
        Axes: the ax with the plot.
    """
    quantiles = np.quantile(results, q=q, axis=1)

    if ax is None:
        _, ax = plt.subplots()

    x = range(1, (len(results) + 1) * points_between_ests, points_between_ests)
    _ = ax.fill_between(
        x,
        quantiles[0],
        quantiles[-1],
        label=f"{q[0] * 100}% to {q[-1] * 100}% Confidence Bound {name}",
        alpha=0.3,
        **kwargs,
    )

    for q_idx in range(1, len(q) - 1):
        _ = ax.plot(x, quantiles[q_idx], **kwargs)

    return ax
