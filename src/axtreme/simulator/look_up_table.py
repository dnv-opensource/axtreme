"""Module for a simulator which finds the closest point in a look up table and returns the value."""

from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy.spatial import KDTree

from axtreme.simulator import Simulator


class LookUpTable(Simulator):
    """Class for a simulator which finds the closest point in a look up table and returns the value.

    If there is multiple possible values for a given point one of them is selected at random.

    To speed this up, a KDTree is used to find the closest point in the look up table.
    However, this functionality can be turned off if desired since it might be faster to not use the KDTree
    if the __call__ method is called very few times and there are very few points.
    """

    def __init__(
        self,
        points: np.ndarray[tuple[int, int], Any],
        values: Sequence[np.ndarray[tuple[int, int], Any]],
        *,
        use_kdtree: bool = True,
        kd_tree_kwargs: dict[str, Any] | None = None,
        seed: int | None = None,
    ) -> None:
        """Initializes the LookUpTable.

        Args:
            points: A numpy array of shape (n_lookup_points, n_input_dims) of points in the input space.
            values: A sequence of with length n_lookup_points of numpy arrays
                - The i-th tensor should contain the values of the function at the i-th point in the input space.
                - Each tensor should have shape (n_possible_values, n_output_dims).
                - The n_possible_values can be different for each point.
            use_kdtree: Whether to use a KDTree to speed up the look up. Default is True.
                - If False, numpy is used to find the closest point.
                - If the __call__ method is called very few times and there are very few points, it might be faster to
                not use the KDTree.
            kd_tree_kwargs: Keyword arguments to pass to the KDTree constructor (scipy.spatial.KDTree for details)
                Default is None.
            seed: The seed for the random number generator. Default is None.
        """
        self.points = points
        self.values = values
        self.random_generator = np.random.default_rng(seed)
        if use_kdtree:
            self.kdtree = KDTree(points, **(kd_tree_kwargs or {}))
            self.lookup = self.lookup_kdtree
        else:
            self.lookup = self.lookup_numpy

    def lookup_numpy(self, x: np.ndarray[tuple[int, int], Any]) -> np.ndarray[tuple[int,], np.dtype[np.int64]]:
        """Find the closest point in the look up table.

        Args:
            x: A tensor of shape (n_points, n_input_dims) of points at which to evaluate the model.

        Returns:
            A tensor of shape (n_points,) of the index of the closest point in the look up table.
        """
        # Find closest point
        dists = np.linalg.norm(self.points[:, None] - x[None, :], axis=-1)
        closest_points = np.argmin(dists, axis=0)

        return closest_points

    def lookup_kdtree(self, x: np.ndarray[tuple[int, int], Any]) -> np.ndarray[tuple[int,], np.dtype[np.int64]]:
        """Find the closest point in the look up table using the KDTree.

        Args:
            x: A tensor of shape (n_points, n_input_dims) of points at which to evaluate the model.

        Returns:
            A tensor of shape (n_points,) of the index of the closest point in the look up table.
        """
        # Find closest point using KDTree
        _, closest_points = self.kdtree.query(x)
        return np.array(closest_points)

    def __call__(
        self, x: np.ndarray[tuple[int, int], Any], n_simulations_per_point: int = 1
    ) -> np.ndarray[tuple[int, int, int], Any]:
        """Evaluate the model at given points.

        Args:
            x: A tensor of shape (N, n_input_dims) of points at which to evaluate the model.
            n_simulations_per_point: The number of simulations to run at each point.

        Returns:
            A tensor of shape (n_points, n_simulations_per_point, n_output_dims) of the model evaluated at the input
            points.
        """
        # Find closest point
        closest_points = self.lookup(x)

        # Select values
        # For each point in closest points, find the values and select n_simulations_per_point of them
        values = np.array(
            [
                self.values[point][
                    self.random_generator.choice(self.values[point].shape[0], n_simulations_per_point, replace=True)
                ]
                for point in closest_points
            ]
        )

        return values
