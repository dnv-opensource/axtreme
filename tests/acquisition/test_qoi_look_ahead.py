# pyright: reportUnnecessaryTypeIgnoreComment=false

# sourcery skip: no-conditionals-in-tests

from collections.abc import Callable
from unittest.mock import patch

import matplotlib.pyplot as plt
import pytest
import torch
from botorch.models import SingleTaskGP
from botorch.models.model import Model
from botorch.optim import optimize_acqf

from axtreme.acquisition.qoi_look_ahead import (
    QoILookAhead,
    average_observational_noise,
    closest_observational_noise,
    conditional_update,
)
from axtreme.plotting.gp_fit import plot_1d_model
from axtreme.qoi.qoi_estimator import QoIEstimator
from axtreme.sampling import NormalIndependentSampler
from axtreme.utils.gradient import is_smooth_1d
from tests.acquisition.helpers import check_posterior, check_same_model_data
from tests.helpers import (
    single_task_fixed_noise_m_1,
    single_task_fixed_noise_m_1_outcome_transform,
    single_task_fixed_noise_m_2,
    single_task_homo_noise_m_1,
)


def test_get_closest_observational_noise():
    new_points = torch.tensor([[0.1, 0.1], [0.16, 0.16], [0.4, 0.4], [0.4, 0.4]])
    train_x = torch.tensor([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]])
    train_yvar = torch.tensor([[0.11], [0.22], [0.33]])

    # Run test
    actual_output = closest_observational_noise(new_points, train_x, train_yvar)

    torch.testing.assert_close(actual_output, torch.tensor([[0.1100], [0.2200], [0.3300], [0.3300]]))


def test_get_average_observation_noise():
    """check the right shaped output is created"""
    new_points = torch.tensor([[0.1, 0.1], [0.16, 0.16], [0.4, 0.4], [0.4, 0.4]])
    # m = 2
    train_yvar = torch.tensor([[0.11, 0.12], [0.22, 0.23], [0.36, 0.37]])

    actual_output = average_observational_noise(new_points, None, train_yvar)

    expected_result = torch.tensor([[0.23, 0.24], [0.23, 0.24], [0.23, 0.24], [0.23, 0.24]])
    torch.testing.assert_close(actual_output, expected_result)


@pytest.mark.parametrize(
    "method",
    [
        (average_observational_noise),
        pytest.param(closest_observational_noise, marks=pytest.mark.xfail(reason="First derivate is not smooth")),
    ],
)
def test_obersvational_noise_getting_is_smooth(
    method: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
) -> None:
    """Check the function getting yvar_new is smooth, which is important for optimisation.

    _get_fantasy_observation_noise using the an underlying helper function to get the yvar_new.
    The helper need to be a smooth function if all optimisation tools are expected to be useable.
    """
    new_points = torch.linspace(0, 1, 100).reshape(-1, 1)

    train_x = torch.tensor([[0.1], [0.3], [0.6], [0.9]])
    train_yvar = torch.tensor([[0.1], [0.9], [0.1], [0.9]])

    yvar_new_function = method(new_points, train_x, train_yvar)

    _ = is_smooth_1d(new_points.flatten(), yvar_new_function.flatten())


class TestQoILookAhead:
    """Unit tests associated with the QoILookAhead class."""

    @pytest.mark.parametrize(
        "b,n,d,m, expected_output",
        [  # Test without batching
            (None, 5, 2, 1, torch.tensor([0, 1, 2, 3, 4])),
            # target dim 2
            (None, 5, 2, 2, torch.tensor([0, 2, 4, 6, 8])),
            # add batch dim
            (2, 5, 2, 2, torch.tensor([[0, 2, 4, 6, 8], [10, 12, 14, 16, 18]])),
        ],
    )
    def test_batch_lookahead_correct_reshape(
        self, b: None | int, n: int, d: int, m: int, expected_output: torch.Tensor
    ):
        """Check that the reshape process used to iterate through lookahead keeps values in the right place.

        Args:
            - b: the batch dimensions.
            - n: the number of points.
            - d: dimension of x points.
            - m: number of targets.
            - expected_output: expected output of the function.
        """
        ### set up the objects
        # Set up the object
        # TODO @ClaasRostock: Not sure if should be using mock/patch libs here. seemed like overkill.
        acqf = QoILookAhead(model=None, qoi_estimator=None)  # type: ignore[arg-type]

        # mock the lookahead function. Simply return the first dimension in y
        def lookahead(x_point: torch.Tensor, y_point: torch.Tensor, yvar_point: torch.Tensor | None) -> torch.Tensor:
            return y_point[0]

        acqf.lookahead = lookahead  # type: ignore[method-assign]

        # Create the inputs
        if b is None:
            # (*b,n,d)
            x_points = torch.ones(n, d)
            # (*b,n,m)
            y_points = torch.arange(n * m).reshape(n, m)
            yvar_points = torch.ones_like(y_points) * 0.1
        else:
            x_points = torch.ones(b, n, d)
            # (*b,n,m)
            y_points = torch.arange(b * n * m).reshape(b, n, m)
            yvar_points = torch.ones_like(y_points) * 0.1

        actual_result = acqf._batch_lookahead(x_points, y_points, yvar_points)

        torch.testing.assert_close(actual_result, expected_output)

    def test_fantasy_observations(self):
        """This is an important function, but it largely just orchestrates other helpers. The main point is ensuring the
        overall shape is correct. This is mainly a simple integration test, except `_get_fantasy_observation_noise` is
        mocked as the shape is the most important parts (the content is still somewhat liable to change)

        The most common case this function is used for.:
        - no gradient
        - m = 2
        - posterior_sampler produces 3 samples

        TODO(sw 2025-4-28): further variation should be tested. This should cover:
            - gradient
            - Single and multi output samplers
            - Multi and single output models
            - fixed and homoskedastic noise models
        """
        # Return shape is (t_batch_,m)
        # (n = 3, m = 1)
        # current approach is obs from the same x point gets the same var. We should check this happens
        x = torch.tensor([[0.11], [0.22], [0.33]])
        n = x.shape[0]
        # (n = 3, m = 2)
        yvar = torch.tensor([[0.1, 0.11], [0.2, 0.22], [0.3, 0.33]])

        n_samples = 4
        sampler = NormalIndependentSampler(sample_shape=torch.Size([n_samples]), seed=0)
        model = single_task_fixed_noise_m_2()
        m = 2

        acquisition = QoILookAhead(
            model=model,
            # qoi_estimator is not used in this test, but is required to instantiate the class
            qoi_estimator=None,  # type: ignore[arg-type]
            sampler=sampler,
        )

        with patch("axtreme.acquisition.qoi_look_ahead._get_fantasy_observation_noise", return_value=yvar):
            x_batch, y_batch, yvar_batch = acquisition.fantasy_observations(x)

        # x expected shape: (n_posterior_samples,n,d) - duplicated across posterior sample dimension
        expected_x_batch = torch.stack([x for _ in range(n_samples)], dim=0)
        torch.testing.assert_close(x_batch, expected_x_batch)

        # y expected shape: (n_posterior_samples,n,m) - all values should be unique
        assert y_batch.shape == (n_samples, n, m)

        # fantasy_observation uses the same yvar for every sample at the same x point
        # yvar expected shape: (n_posterior_samples,n,m) - duplicated across posterior sample dim
        expected_yvar_batch = torch.stack([yvar for _ in range(n_samples)], dim=0)
        torch.testing.assert_close(yvar_batch, expected_yvar_batch)

    def test_aggregate_lookahead_results(self):
        """Check the sampler mean is used if provided (e.g for cases such a UTsampler)"""

        class DummySampler:
            """Custom sampler that has a non-standard mean function."""

            def mean(self, transformed_points: torch.Tensor, dim: int = 0) -> torch.Tensor:
                return transformed_points[0]

        acquisition = QoILookAhead(
            model=None,  # type: ignore[arg-type] # Require to instantiate, but not used.
            qoi_estimator=None,  # type: ignore[arg-type] # Require to instantiate, but not used.
            sampler=DummySampler(),  # type: ignore[arg-type]
        )

        x = torch.tensor([[1], [2], [3]])
        # checks the non-standard mean was used rather than default mean
        assert acquisition._aggregate_lookahead_results(x) == torch.tensor([1])


# This warning is due to train_y not being standardised. Standarisation can improve quality of fit.
# This is not important for this test.
@pytest.mark.filterwarnings(
    "ignore : Input data is not standardized : botorch.exceptions.InputDataWarning : botorch.models"
)
def test_conditional_update():
    """Test connection to underling `condition_on_observation`.

    NOTE: `condition_on_observation` is tested more extensively in `test_qoi_look_ahead_external.py`
    """
    train_x = torch.tensor([[0.1], [0.2], [0.3]])
    train_y = torch.sin(train_x)
    train_yvar = torch.ones_like(train_y) * 0.1

    model_original = SingleTaskGP(train_x, train_y, train_yvar)

    # Generate new training data to use
    new_x = torch.tensor([[0.5]])  # (n,d)
    new_y = torch.tensor([[0.1]])
    new_yvar = torch.ones_like(new_y) * 0.1

    # New conditioned model
    model_conditioned = conditional_update(model_original, X=new_x, Y=new_y, observation_noise=new_yvar)

    # Expected model
    x_total = torch.vstack([train_x, new_x])  # (n+1,d=1)
    y_total = torch.vstack([train_y, new_y])  # (n+1,n_targets)
    yvar_total = torch.vstack([train_yvar, new_yvar])  # (n+1,n_targets)
    model_expected = SingleTaskGP(x_total, y_total, yvar_total)

    # Check the internal data is the same
    check_same_model_data(model_actual=model_conditioned, model_expected=model_expected)
    # check that they both make the same preditiontion
    x = torch.tensor([[0.4], [0.7]])
    expected_posterior = model_expected.posterior(x)
    conditioned_posterior = model_conditioned.posterior(x)
    check_posterior(conditioned_posterior, expected_posterior)


@pytest.mark.parametrize(
    "model, x, y, observation_noise, error_message",
    [
        pytest.param(
            single_task_homo_noise_m_1(),
            None,
            None,
            torch.tensor([0.1]),
            "observation_noise is not supported",
            id="Homoskedatic model recieved observation noise",
        ),
        pytest.param(
            single_task_fixed_noise_m_1(),
            None,
            None,
            None,
            "requires observation_noise",
            id="Fixed noise model missing observation noise",
        ),
        pytest.param(
            single_task_homo_noise_m_1(),
            torch.ones(1, 1, 1),
            torch.ones(1, 1, 1),
            None,
            "batch shape",
            id="Batched data doesn't match non-batched model",
        ),
    ],
)
def test_conditional_update_exception_checking(
    model: SingleTaskGP,
    x: None | torch.Tensor,
    y: None | torch.Tensor,
    observation_noise: None | torch.Tensor,
    error_message: str,
):
    """Check the differnet warning conditions are approapriately checked.

    Args:
        model: the model being updated.
        x: The additional points to add. None here if not involved with triggering the error
        y: The additional points to add. None here if not involved with triggering the error
        observation_noise: yvar of the additional points to add. None here if not involved with triggering the error
        error_message: substring expected in the error message.
    """
    with pytest.raises(Exception, match=error_message):
        _ = conditional_update(model, X=x, Y=y, observation_noise=observation_noise)  # type: ignore[arg-type]


def test_conditional_update_warning_checking_input_output_transforms():
    model = single_task_fixed_noise_m_1_outcome_transform()
    # Note: this is not important, just to prevent other error from failing so we can get to the one we want
    data = torch.tensor([[0.1]])

    with pytest.warns(UserWarning, match="outcome transforms"):
        _ = conditional_update(model, X=data, Y=data, observation_noise=data)


def test_forward_grad():
    r"""Test the acquisition function correctly handles gradient information.

    This is currently not tested so function should raise not yet implemented error.

    Todo:
    - Make simple test like `test_optimise_dumb_qoi` but with gradient turned on.
    - make more sophisticated tests:
        - These might cover:
            - what happens in the optimisation when there is noise in the QoI.
    - Existing material that can serve as a base.
        -`explorations\qoi_look_ahead_gradient.ipynb` to build gradient tests. This covers:
            - ploting of the acqusition function gradient at a grid of points.
            - Running with Ax with and withotu gradient to see the difference.
        - `explorations\dev_ks\qoi_acquisition\qoi_acquisition_first_trial.ipynb`
            - rough example optimising a problem with gradient through Ax.

    """
    acqf = QoILookAhead(model=None, qoi_estimator=None)  # type: ignore[arg-type]
    x = torch.ones(1, 1, 1, requires_grad=True)

    with pytest.raises(NotImplementedError):
        acqf(x)


# Would be nice to test GPU, but there is not flag so this is difficult.
# The underling module has no params to check either.


@pytest.mark.integration
# This warning is due to train_y not being standardised. Standardisation can improve quality of fit.
# This is not important for this test.
@pytest.mark.filterwarnings(
    "ignore : Input data is not standardized : botorch.exceptions.InputDataWarning : botorch.models"
)
def test_acquisition_function_is_smooth(*, visual_inspect: bool = False):  # noqa: PT028
    """Assuming the requirement detailed in the class docs are met, the acquisition function produced should be
    smooth.
    """

    train_x = torch.tensor([[0.1], [0.5], [0.9]])
    train_y = torch.zeros_like(train_x)
    train_yvar = torch.tensor([[0.9], [0.1], [0.9]])
    model = SingleTaskGP(train_x, train_y, train_yvar)

    if visual_inspect:
        _ = plot_1d_model(model)
        plt.pause(5)  # required to be able to see the plot when running in pytest

    class DummyQoI(QoIEstimator):
        """Simply QoI that only cares about reducing the variance at one point.

        Only cares about the variance at `x=.5` The var method is then just a pass through of the value returned by x.

        The output doesn't truly conform with the QoI interface, but is sufficient for this test.
        """

        def __call__(self, model: Model) -> torch.Tensor:
            x = torch.tensor([[0.5]])
            with torch.no_grad():
                posterior = model.posterior(x)

            return posterior.variance.flatten()  # type: ignore  # noqa: PGH003

        def var(self, x: torch.Tensor) -> torch.Tensor:
            return x[0]

    dummy_qoi = DummyQoI()
    acqf = QoILookAhead(model, qoi_estimator=dummy_qoi)

    x = torch.linspace(0, 1, 200).reshape(-1, 1, 1)
    scores = acqf(x)

    if visual_inspect:
        _ = plt.plot(x.flatten(), scores.flatten())
        plt.pause(5)

    _ = is_smooth_1d(x.flatten(), scores.flatten())


# This warning is due to train_y not being standardised. Standardisation can improve quality of fit.
# This is not important for this test.
@pytest.mark.filterwarnings(
    "ignore : Input data is not standardized : botorch.exceptions.InputDataWarning : botorch.models"
)
@pytest.mark.integration
# TODO(sw 2024-11-27): This should be updated to use the default params registered in qoi_look_ahead.py
def test_optimise_dumb_qoi(*, visual_inspect: bool = False):  # noqa: PT028
    """Make a QoI that only cares about reducing the variance as a single point.

    Check that the optimisation can find that point.

    Args:
        visual_inspect: Flag to create visualisation of test for manual inspection.

    Note:
        We could also test more complicated surfaces and show they work. Testing if optimiser setting work on a
        realistic problem surface is consider system testing. Testing if optimisation setting work on arbitrary
        problem surface should be in unittests of the optimiser (with arbitrary functions) rather than here.
    """
    train_x = torch.tensor([[0.1], [0.9]])
    train_y = torch.zeros_like(train_x)
    train_yvar = torch.ones_like(train_y) * 0.1
    model = SingleTaskGP(train_x, train_y, train_yvar)

    if visual_inspect:
        _ = plot_1d_model(model)
        plt.pause(5)  # required to be able to see the plot when running in pytest

    class DummyQoI(QoIEstimator):
        """Simply QoI that only cares about reducing the variance at one point.

        Only cares about the variance at `x=.5` The var method is then just a pass through of the value returned by x.

        The output doesn't truely conform with the QoI interface, but is suffecient for this test.
        """

        def __call__(self, model: Model) -> torch.Tensor:
            x = torch.tensor([[0.5]])
            with torch.no_grad():
                posterior = model.posterior(x)

            return posterior.variance.flatten()  # type: ignore  # noqa: PGH003

        def var(self, x: torch.Tensor) -> torch.Tensor:
            return x[0]

    dummy_qoi = DummyQoI()
    acqf = QoILookAhead(model, qoi_estimator=dummy_qoi)

    candidate, result = optimize_acqf(
        acqf,
        bounds=torch.tensor([[0.0], [1.0]]),
        q=1,
        # This is how many different start location will be tried by the optimiser
        num_restarts=5,
        raw_samples=100,
        # Key parameter to control if optimisation should use gradient from the acquisiton fucntion
        options={"with_grad": False},  # True by default
    )

    # Could run the optimisation longer if we require it to be more precise. This is considered approapriate for a test.
    torch.testing.assert_close(candidate, torch.tensor([[0.5]]), rtol=0, atol=1e-3)
