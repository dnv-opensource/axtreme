import numpy as np
import torch
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize

from axtreme.acquisition.optimal_qoi_look_ahead import OptimalQoILookAhead
from axtreme.evaluation import SimulationPointResults


def test_fantasy_observation():
    """Checks values are appropriately transformed from model space to problem space and back.

    Using arbitrary transforms, transfer input to problem space, run evaluation function, and transfer results back. The
    normal distribution given by the evaluation function is shifted back to model space and mean and variance are
    checked by doing a manual calculation.
    """

    # Set up:
    class DummyEvaluationFunction:
        """Dummy evaluation function to test the fantasy_observations method.

        Minimal version of EvaluationFunction just implementing the methods required.

        This is define in the problem space.
            - input x: [100,102]
            - output function:
                - y = x + 10
                - yvar = x - 100 + 1
        """

        def evaluate(self, x: np.ndarray[tuple[int], np.dtype[np.float64]]) -> SimulationPointResults:
            return SimulationPointResults(metric_names=["junk"], means=x + 10, cov=x.reshape(1, -1) - 100 + 1)

    # Creates unit cube
    input_transform = Normalize(d=1, bounds=torch.Tensor([[100], [102]]))
    # Create a transform. Note: it doesn't matter what it is, just that we can tell if its been correctly applied.
    outcome_transform = _make_standarize_transform(
        desired_mean=torch.tensor([105.0]),
        desired_std=torch.tensor([2.0]),
    )

    acquisition = OptimalQoILookAhead(
        model=None,  # type: ignore[arg-type] # Not required for method under test.
        qoi_estimator=None,  # type: ignore[arg-type] # Not required for method under test.
        evaluation_function=DummyEvaluationFunction(),  # type: ignore[arg-type]
        input_transform=input_transform,
        outcome_transform=outcome_transform,
    )

    # Run the function
    x_model_space = torch.tensor([[0.1], [0.5], [0.9]])
    _, y_model_actual, yvar_model_actual = acquisition.fantasy_observations(x_model_space)

    # Manually perform calc to get the expected result
    x_problem = x_model_space * 2 + 100
    y_problem = x_problem + 10
    yvar_problem = x_problem - 100 + 1
    # We consider the EvaluationFunction to provide the distribution of the value in problem space. We then need to
    # transform this model to model space.
    expected_y_model, expected_yvar_model = _manual_conversion_problem_to_model(
        y_problem, yvar_problem, outcome_transform
    )

    torch.testing.assert_close(y_model_actual, expected_y_model)
    torch.testing.assert_close(yvar_model_actual, expected_yvar_model)


def _make_standarize_transform(desired_mean: torch.Tensor, desired_std: torch.Tensor) -> Standardize:
    """Small helper to create a standardize transform.

    Rather than set attribute manually, fit to data so internals are set.

    Args:
        desired_mean(m,): The mean your transform should have. Length determines the number of output dimensions.
        desired_std: The standard deviation of the distribution to be transformed.
    """
    x = torch.tensor([desired_mean - desired_std, desired_mean, desired_mean + desired_std]).reshape(3, -1)
    transform = Standardize(m=len(desired_mean))
    transform(x)
    _ = transform.eval()

    torch.testing.assert_close(transform.means.flatten(), desired_mean)
    torch.testing.assert_close(transform.stdvs.flatten(), desired_std)
    return transform


def _manual_conversion_problem_to_model(
    mean_problem: torch.Tensor, var_problem: torch.Tensor, outcome_transform: Standardize
):
    """Small helper to document the conversion logic in one place

    Manual calculation for shifting distribtuions from the problem space to the model space.

    Args:
        mean_problem: The mean of a distribution in the problem space.
        var_problem: The variance of a distribution in the problem space.
        outcome_transform: The transformation to shift values/distributions from the problem space to the model space.

    Returns:
        mean_model: The mean of the distribution in the model space.
        var_model: The variance of the distribution in the model space.

    Formulas:
        - mean_problem = a * mean_model + b
            - Where a is the scale, b is location.
            - a = outcome_transform.stdvs, b = outcome_transform.means
        - variance_problem = a^2 * variance_model
            -  where a is scale factor
            - a = outcome_transform.stdvs, b = outcome_transform.means

    TODO:
        - Provide a clean write of the maths here.
    """

    mean_model = (mean_problem - outcome_transform.means) / outcome_transform.stdvs
    var_model = var_problem / (outcome_transform.stdvs**2)

    return mean_model, var_model
