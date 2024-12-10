"""Helper functions for creating vaid simulators."""

import inspect
from collections.abc import Callable

import numpy as np

from axtreme.simulator.base import Simulator


def simulator_from_func(
    func: Callable[
        [np.ndarray[tuple[int, int], np.dtype[np.float64]]], np.ndarray[tuple[int, int], np.dtype[np.float64]]
    ],
) -> Simulator:
    """Adds `n_simulations_per_point` functionality to a a function that takes x inputs, and return simulation values.

    Args:
        func: function which accepts (n,d) x input, and returns (n, m) y output (one response per x point).

    Returns:
        Function conforming to Simultator interface. Output will always be 3 dimensional
        (n_points, n_simulations_per_point, n_targets).
    """

    def simulator(
        x: np.ndarray[tuple[int, int], np.dtype[np.float64]], n_simulations_per_point: int = 1
    ) -> np.ndarray[tuple[int, int, int], np.dtype[np.float64]]:
        # repeat x to get num_simulations_per_statebounds
        x_expanded = np.repeat(x, n_simulations_per_point, axis=0)

        # evaluate the function
        y = func(x_expanded)

        # reshape y to desired shape (n, n_simulations_per_point, d)
        return y.reshape(x.shape[0], n_simulations_per_point, -1)

    return simulator


def is_valid_simulator(instance: object, *, verbose: bool = False) -> bool:  # noqa: C901, PLR0911
    """Checks that `instance` conforms to Simulator protocol AND signature definition.

    This test only works if the instance has type definitions.

    Note:
        By default, `isinstance` with `@runtime_checkable` "will check only the presence of the required methods not
        their type signatures!". See `@runtime_checkable` docs for details.

    Args:
        instance: will check if this is a valid Simulator.
        verbose: If true will provide information regarding why it does not conform.

    Returns:
        bool: True if methods and signatures match

    Note:
        - Briefly looked at override the __instancecheck__ so isinstance would acheive this result
        - Apprears __instancecheck__ doesn't get called, would need to investigate why to proceed
        - Adding this behaviour to protocol would cause it to depart from standard protocol behaviour
    """
    if not isinstance(instance, Simulator):
        if verbose:
            print("Instance does not have the same methods as `Simulator`") if verbose else None  # noqa: T201
        return False

    # Specific check regarding which parameters need the same name, defaults, and type annotations.
    # For functions can inspect them directly, for classes need to inspect the __call__ method.
    instance_sig = inspect.signature(instance) if inspect.isfunction(instance) else inspect.signature(instance.__call__)

    sim_sig = inspect.signature(Simulator.__call__)

    # check everything in the simulator signature is in the instance signature
    for sim_param, sim_info in sim_sig.parameters.items():
        if sim_param != "self":
            # Check param is available in instance
            if sim_param not in instance_sig.parameters:
                print(f"Input is missing parameter '{sim_param}'") if verbose else None  # noqa: T201
                return False
            # check the type is correct:
            if sim_info.annotation != instance_sig.parameters[sim_param].annotation:
                print(f"Parameter {sim_param} has incorrect type.") if verbose else None  # noqa: T201
                return False
            # If a default is provided as part of sim, instance should have one too
            if (sim_info.default is not inspect.Parameter.empty) and (
                instance_sig.parameters[sim_param].default is inspect.Parameter.empty
            ):
                print(f"Parameter {sim_param} is missing  a default") if verbose else None  # noqa: T201
                return False

    # check any extras in the instance signature has a default.
    for instance_param, instance_info in instance_sig.parameters.items():
        if instance_param != "self" and instance_param not in sim_sig.parameters:  # noqa: SIM102
            if instance_info.default is inspect.Parameter.empty:
                print(  # noqa: T201
                    f"{instance_param} is an additional parameter to Simulator interface, and has no default value."
                ) if verbose else None
                return False

    if instance_sig.return_annotation != sim_sig.return_annotation:
        if verbose:
            print(  # noqa: T201
                f"Parameter return types are different: instance: {instance_sig.return_annotation}"
                f" Expected: {sim_sig.return_annotation}"
            )
        return False

    return True
