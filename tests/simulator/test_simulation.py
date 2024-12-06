import numpy as np
from numpy.typing import NDArray

from axtreme.simulator import utils


def test_is_valid_simulator():
    def foo(
        x: np.ndarray[tuple[int, int], np.dtype[np.float64]], n_simulations_per_point: int = 1
    ) -> np.ndarray[tuple[int, int, int], np.dtype[np.float64]]:
        raise NotImplementedError

    assert utils.is_valid_simulator(foo, verbose=True)


def test_is_valid_simulator_incorrect_params():
    def foo(
        x: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    ) -> np.ndarray[tuple[int, int, int], np.dtype[np.float64]]:
        raise NotImplementedError

    assert utils.is_valid_simulator(foo) is False


def test_is_valid_simulator_incorrect_types():
    def foo(
        x: np.ndarray[tuple[int, int], np.dtype[np.float64]], n_simulations_per_point: str
    ) -> np.ndarray[tuple[int, int, int], np.dtype[np.float64]]:
        raise NotImplementedError

    assert utils.is_valid_simulator(foo) is False


def test_is_valid_simulator_incorrect_missing_default():
    def foo(
        x: np.ndarray[tuple[int, int], np.dtype[np.float64]], n_simulations_per_point: int
    ) -> np.ndarray[tuple[int, int, int], np.dtype[np.float64]]:
        raise NotImplementedError

    assert utils.is_valid_simulator(foo) is False


def test_is_valid_simulator_incorrect_return():
    def foo(x: np.ndarray[tuple[int, int], np.dtype[np.float64]], n_simulations_per_point: int = 1) -> bool:
        raise NotImplementedError

    assert utils.is_valid_simulator(foo) is False


def test_is_valid_simulator_allow_custom_defaults():
    def foo(
        x: np.ndarray[tuple[int, int], np.dtype[np.float64]], n_simulations_per_point: int = 10
    ) -> np.ndarray[tuple[int, int, int], np.dtype[np.float64]]:
        raise NotImplementedError

    assert utils.is_valid_simulator(foo)


def test_simulator_from_func_adds_n_simulations_per_point_correctly():
    def dummy_function(x: NDArray[np.float64]) -> NDArray[np.float64]:
        return x + 1

    simulator = utils.simulator_from_func(dummy_function)

    x = np.array([1, 10]).reshape(-1, 1)
    expected_output = np.array([2, 2, 11, 11]).reshape(2, -1, 1)

    output = simulator(x, n_simulations_per_point=2)

    assert np.allclose(output, expected_output)
