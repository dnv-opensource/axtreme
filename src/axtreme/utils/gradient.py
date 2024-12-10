"""Helper for gradient assessment.

NOTE: currently just used as a helper in tests. Could move their if we do not consider it useful to users
"""

import matplotlib.pyplot as plt
import torch
from matplotlib.figure import Figure


def is_smooth_1d(
    x: torch.Tensor,
    y: torch.Tensor,
    d1_threshold: float = 3.0,
    d2_threshold: float = 150,
    *,
    plot: bool = False,
    test: bool = True,
) -> None | Figure:
    """Helper to warn if 1d function is likely not smooth (twice differntiable).

    Args:
        x: (n,) points representing the x (input) values of a function
        y: (n,) points representing the y (output) values of a function
        d1_threshold: Maximum step size allowed in 1st derivative function to be considered smooth.
        d2_threshold: Maximum step size allowed in 1st derivative function to be considered smooth.
        plot: If true, will plot the first and second derivative functions.
        test: is assert statments should be run

    Details
        Smoothness: (defined up to K, the Kth derivative you can take that give a contious funciton over domain)
            - C_0: set of continous functions
            - C_1 (once differentiable):
                - C_0 can be differentiated
                - AND the resulting function is continuous (No steps or holes)
            - ...

        We want C_2 (twice differentiable) for optimisation. As such here we test.
            - if f'(x) is continous.
                - Check if there are not step = not big changes between points.
                - Knowing the appropriate step threshold is hard, the gradient is easier to think about.
                    - Easy way to do this is check if max slope is not exceeded (derivate 2nd)
            - f''(x) is continous.
                - Check if there are not step = not big changes between points
                - Easy way to do this is check if max slope is not exceeded (derivate 2nd)

    NOTE: The thresholds have been set heuristically. It is recomended to plot your function and establish smoothness,
    the use this for regression testing once suitable values have been determined.

    NOTE: The smaller the step size, the better the estimate of gradient (large and small). See y = sim(50 * x)` with
    torch.linspace(0,1,100)` and `torch.linspace(0,1,1000)`
    """
    step_size = torch.diff(x)

    first_derivative = torch.diff(y) / step_size

    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        _ = axes[0].set_ylabel("1st derivative")
        _ = axes[0].scatter(x[:-1], first_derivative, c="blue", s=2, label="1st derivative")
        _ = axes[0].legend()
        _ = axes[0].set_title("Derivative")

        _ = axes[1].set_title("Derivative Step change")
        _ = axes[1].set_ylabel("1st derivative")
        _ = axes[1].scatter(x[:-2], torch.diff(first_derivative), c="blue", s=2, label="1st derivative")

    d1 = torch.diff(first_derivative).abs() < d1_threshold

    if test:
        msg = (
            f"First derivate is not smooth at indexes {torch.nonzero(~d1).flatten()}.\n These indexes represent the the"
            " 0 based index of 'point_l' in the group of points 'point_l -- point_c -- point_r' in which"
            " the gradient has a step change."
        )
        assert d1.all(), msg

    # each time diff is called, the last element has no next value to calc the diff against
    # as a result that item is dropped
    second_derivative = torch.diff(first_derivative) / step_size[:-1]

    if plot:
        ax1_twin = axes[0].twinx()
        _ = ax1_twin.scatter(x[:-2], second_derivative, c="orange", s=2, label="2nd derivative")
        _ = ax1_twin.set_ylabel("2nd derivative")
        _ = ax1_twin.legend()

        ax2_twin = axes[1].twinx()
        _ = ax2_twin.set_ylabel("2nd derivative")
        _ = ax2_twin.scatter(x[:-3], torch.diff(second_derivative), c="orange", s=2, label="2nd derivative")
        ax2_twin.legend()
        plt.tight_layout()

    d2 = torch.diff(second_derivative).abs() < d2_threshold

    if test:
        assert d2.all(), f"Second derivate is not smooth at indexs {torch.nonzero(~d2).flatten()}"

    return fig if plot else None
