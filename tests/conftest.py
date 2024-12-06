# sourcery skip: dont-import-test-modules

import logging
import os
import warnings
from pathlib import Path
from shutil import rmtree

import gpytorch
import matplotlib.pyplot as plt
import pytest
import torch
import torch.cuda
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP

from axtreme.plotting.gp_fit import plot_1d_model
from tests import helpers

CUDA_AVAILABLE: bool = torch.cuda.is_available()

TORCH_DEVICES: list[str] = ["cuda", "cpu"] if CUDA_AVAILABLE else ["cpu"]


@pytest.fixture(scope="package", autouse=True)
def chdir() -> None:
    """
    Fixture that changes the current working directory to the 'test_working_directory' folder.
    This fixture is automatically used for the entire package.
    """
    os.chdir(Path(__file__).parent.absolute() / "test_working_directory")


# @pytest.fixture(scope="class", autouse=True)
# def reset_torch_default_device() -> None:
#     torch.set_default_device("cpu")


@pytest.fixture(scope="class", params=TORCH_DEVICES)
def vary_torch_default_device(request: pytest.FixtureRequest):
    torch.set_default_device(request.param)
    yield
    torch.set_default_device("cpu")  # reset to default device after test


@pytest.fixture(scope="package", autouse=True)
def test_dir() -> Path:
    """
    Fixture that returns the absolute path of the directory containing the current file.
    This fixture is automatically used for the entire package.
    """
    return Path(__file__).parent.absolute()


output_dirs = [
    "results",
    "data",
]
output_files = [
    "*test*.pdf",
]


@pytest.fixture(autouse=True)
def default_setup_and_teardown():
    """
    Fixture that performs setup and teardown actions before and after each test function.
    It removes the output directories and files specified in 'output_dirs' and 'output_files' lists.
    """
    _remove_output_dirs_and_files()
    yield
    _remove_output_dirs_and_files()


def _remove_output_dirs_and_files() -> None:
    """
    Helper function that removes the output directories and files specified in 'output_dirs' and 'output_files' lists.
    """
    for folder in output_dirs:
        rmtree(folder, ignore_errors=True)
    for pattern in output_files:
        for file in Path.cwd().glob(pattern):
            _file = Path(file)
            _file.unlink(missing_ok=True)


@pytest.fixture(autouse=True)
def setup_logging(caplog: pytest.LogCaptureFixture) -> None:
    """
    Fixture that sets up logging for each test function.
    It sets the log level to 'INFO' and clears the log capture.
    """
    caplog.set_level("INFO")
    caplog.clear()


@pytest.fixture(autouse=True)
def logger() -> logging.Logger:
    """Fixture that returns the logger object."""
    return logging.getLogger()


@pytest.fixture(
    params=[
        helpers.single_task_homo_noise_m_1(),
        helpers.single_task_homo_noise_m_2(),
        helpers.single_task_fixed_noise_m_1(),
        helpers.single_task_fixed_noise_m_2(),
        helpers.single_task_homo_noise_m_2_b3(),
    ],
    # provides naming to easily identify model used
    ids=[
        "single_task_homo_noise_m_1",
        "single_task_homo_noise_m_2",
        "single_task_fixed_noise_m_1",
        "single_task_fixed_noise_m_2",
        "single_task_homo_noise_m_2_b3",
    ],
)
def botorch_models(request: pytest.FixtureRequest):
    """Provides a generic set of model to be tested against.

    This is useful when functionality should be invariant to specific type of model.

    NOTE: Its not possible to paramterise fixture so we do this instead.
    """
    return request.param


@pytest.fixture(scope="session")
def model_singletaskgp_d1_t1(*, visual_inspect: bool = False):
    """Creates a basic gp with 1d input and 1d output."""

    def y_func(x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x * 10)

    train_x = torch.tensor([[0.1], [0.5], [0.9], [1.0]])
    train_y = y_func(train_x)

    # Ignore the warning about centralise data, we have inspected it and are fine with it.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*Please consider scaling the input.*")

        model = SingleTaskGP(train_X=train_x, train_Y=train_y, train_Yvar=torch.ones_like(train_y) * 0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
        _ = fit_gpytorch_mll(mll)

    if visual_inspect:
        _ = plot_1d_model(model)

    return model


@pytest.fixture(scope="session")
def model_singletaskgp_d1_t2(*, visual_inspect: bool = False):
    """Creates a basic gp with 1d input and 2d output.

    This output can be used to represent location and scale modeling. The functions have not been choosen in any
    particulat way, except the 2nd function (scale) is larget than 0.
    """

    def target1(x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x * 10)

    def target2(x: torch.Tensor) -> torch.Tensor:
        return 0.5 * torch.cos(x * 10) + 0.8

    def y_func(x: torch.Tensor) -> torch.Tensor:
        return torch.concat([target1(x), target2(x)], dim=-1)

    train_x = torch.tensor([[0.1], [0.5], [0.9], [1.0]])
    train_y = y_func(train_x)

    # Ignore the warning about centralise data, we have inspected it and are fine with it.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*Please consider scaling the input.*")

        model = SingleTaskGP(train_X=train_x, train_Y=train_y, train_Yvar=torch.ones_like(train_y) * 0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
        _ = fit_gpytorch_mll(mll)

    if visual_inspect:
        _, ax = plt.subplots(1, 1, figsize=(6, 4))
        x = torch.linspace(0, 1, 100).reshape(-1, 1)
        _ = ax.plot(x, target1(x), label="target1")
        _ = ax.plot(x, target2(x), label="target2")
        _ = plot_1d_model(model, ax=ax)
        _ = ax.legend()

    return model
