"""3D histogram plotting function.

Plotly does not have a plotly.graph_objects.Histogram3d object, this is an attempt to remedy that.
The closest is plotly.graph_objects.Histogram2d, but that creates a 2D heatmap.

See Also:
- https://github.com/serge-tochilov/barchart3d-plotly
- https://github.com/AymericFerreira/Plotly_barchart3D
- https://community.plotly.com/t/how-to-do-a-3d-bar-chart-if-possible/32287/5
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import plotly.graph_objects as go


def histogram3d(  # noqa: PLR0913
    samples: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    *,
    n_bins: int = 20,
    bounds: Sequence[tuple[float, float]] | None = None,
    density: bool = True,
    flatshading: bool = True,
    show: bool = False,
    mesh3d_kwargs: dict[str, Any] | None = None,
    layout_kwargs: dict[str, Any] | None = None,
) -> go.Figure:
    """Generate a 3D figure of a 2D histogram of the input samples.

    Computes a 2D histogram that is then plotted in 3D as a 3D bar chart (confusing naming...).

    Args:
        samples: An array of shape (n_samples, 2) of samples to compute the histogram of.
        n_bins: The number of bins to use in each dimension.
        bounds: An array of shape (2, 2) of the bounds of the histogram: ((x_min, x_max), (y_min, y_max)). If None, the
            bounds are handled by numpy.histogramdd.
        density: If True, the histogram is normalized to form a probability density function.
        flatshading: If True, the 3D bars are shaded smoothly.
        show: If True, the figure is displayed.
        mesh3d_kwargs: A dictionary of keyword arguments to pass to the plotly Mesh3d object.
        layout_kwargs: A dictionary of keyword arguments to pass to the update_layout method of the figure.

    Returns:
        A plotly figure of the 3D histogram. If show is True, the figure is also displayed.
    """
    # set default values
    if mesh3d_kwargs is None:
        mesh3d_kwargs = {}
    if layout_kwargs is None:
        layout_kwargs = {}

    if bounds is None:
        bounds = [
            (float(samples[:, 0].min()), float(samples[:, 0].max())),
            (float(samples[:, 1].min()), float(samples[:, 1].max())),
        ]

    # compute the histogram
    hist, edges = np.histogramdd(
        samples,
        bins=n_bins,
        range=bounds,
        density=density,
    )
    x_edges = edges[0]
    y_edges = edges[1]

    # construct the 3D bars manually
    fig = go.Figure()

    # add bars for each bin
    for i in range(n_bins):
        for j in range(n_bins):
            z_value = hist[i, j]
            if z_value > 0:  # only plot non-empty bins
                x_min = x_edges[i]
                x_max = x_edges[i + 1]
                y_min = y_edges[j]
                y_max = y_edges[j + 1]

                _ = fig.add_trace(
                    go.Mesh3d(
                        x=[x_min, x_min, x_max, x_max, x_min, x_min, x_max, x_max],
                        y=[y_min, y_max, y_max, y_min, y_min, y_max, y_max, y_min],
                        z=[0, 0, 0, 0, z_value, z_value, z_value, z_value],
                        alphahull=0,
                        **{
                            "intensity": [0, 0, 0, 0, z_value, z_value, z_value, z_value],
                            "flatshading": flatshading,
                            "coloaxtremeis": "coloaxtremeis",
                            **mesh3d_kwargs,
                        },
                    ),
                )

    _ = fig.update_layout(
        **{
            "scene": {
                "xaxis": {"title": "X1", "range": bounds[0]},
                "yaxis": {"title": "X2", "range": bounds[1]},
                "zaxis": {"title": "Density" if density else "Count"},
            },
            **layout_kwargs,
        },
    )

    if show:
        fig.show()

    return fig


def histogram_surface3d(
    samples: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    *,
    n_bins: int = 20,
    bounds: Sequence[tuple[float, float]] | None = None,
    density: bool = True,
    show: bool = False,
    surface3d_kwargs: dict[str, Any] | None = None,
    layout_kwargs: dict[str, Any] | None = None,
) -> go.Figure:
    """Generate a 3D figure of a 2D histogram of the input samples.

    Computes a 2D histogram that is then plotted in 3D as a surface plot.

    Args:
        samples: An array of shape (n_samples, 2) of samples to compute the histogram of.
        n_bins: The number of bins to use in each dimension.
        bounds: An array of shape (2, 2) of the bounds of the histogram: ((x_min, x_max), (y_min, y_max)). If None, the
            bounds are handled by numpy.histogramdd.
        density: If True, the histogram is normalized to form a probability density function.
        show: If True, the figure is displayed.
        surface3d_kwargs: A dictionary of keyword arguments to pass to the plotly Surface3d object.
        layout_kwargs: A dictionary of keyword arguments to pass to the update_layout method of the figure.

    Returns:
        A plotly figure of the 3D histogram. If show is True, the figure is also displayed.
    """
    # set default values
    if surface3d_kwargs is None:
        surface3d_kwargs = {}
    if layout_kwargs is None:
        layout_kwargs = {}

    if bounds is None:
        bounds = [
            (float(samples[:, 0].min()), float(samples[:, 0].max())),
            (float(samples[:, 1].min()), float(samples[:, 1].max())),
        ]

    # compute the histogram
    hist, edges = np.histogramdd(
        samples,
        bins=n_bins,
        range=bounds,
        density=density,
    )
    x_edges = edges[0]
    y_edges = edges[1]

    fig = go.Figure(
        data=[
            go.Surface(
                x=x_edges,
                y=y_edges,
                z=hist.T,
                **surface3d_kwargs,
            ),
        ],
    )

    _ = fig.update_layout(
        **{
            "scene": {
                "xaxis": {"title": "X1", "range": bounds[0]},
                "yaxis": {"title": "X2", "range": bounds[1]},
                "zaxis": {"title": "Density" if density else "Count"},
            },
            **layout_kwargs,
        },
    )

    if show:
        fig.show()

    return fig
