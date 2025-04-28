"""Adaptation of QoILookAhead that directly uses the simulator when looking ahead."""

from typing import Any

import torch
from ax.models.torch.botorch_modular.optimizer_argparse import _argparse_base, optimizer_argparse
from botorch.acquisition.input_constructors import acqf_input_constructor
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform

from axtreme.acquisition.qoi_look_ahead import QoILookAhead
from axtreme.evaluation import EvaluationFunction
from axtreme.qoi.qoi_estimator import QoIEstimator


class OptimalQoILookAhead(QoILookAhead):
    """Identical to QoILookAhead, but uses the simulator directly to calculate response values.

    This is used in testing and development on toy problems to determine the optimal DOE performance.
    This acquisition function is considered optimal as it directly calculates the QoI at a candidate point x, using
    the true response that will be received by running the experiment at that point.

    Note:
        - if the result of running an experiment is stochastic, then this function may get unusual samples and not
          produce optimal DOE.
        - GP hyperparameters are not refit in this function. If the DOE is evaluated with a GP that is refit after every
          experiment, this method may not choose the optimal point.

    This is done through the EvaluationFunction which orchestrates the fitting of the targets. With `axtreme`
    experiments this can typically be found `experiment.runner.evaluation_function`.
    """

    def __init__(
        self,
        model: SingleTaskGP,
        qoi_estimator: QoIEstimator,
        evaluation_function: EvaluationFunction,
        input_transform: InputTransform,
        outcome_transform: OutcomeTransform,
    ) -> None:
        """Instantiate the OptimalQoILookAhead acquisition function.

        Args:
            model: The model being used at this round of DOE.
            qoi_estimator: The qoi estimator being optimised.
            evaluation_function: This orchestrates the fitting of the targets (get the response from the simulator).
              Within `axtreme` experiments this is typically  found `experiment.runner.evaluation_function`.
            input_transform: The input transform for the model provided. Takes input from problem space to model space.
            outcome_transform: The outcome transform for the model provided. Takes responses from problem space to model
              space.
        """
        super().__init__(model, qoi_estimator)
        self.evaluation_function = evaluation_function
        self.input_transform = input_transform
        self.outcome_transform = outcome_transform

    def fantasy_observations(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Takes input x in model space, and returns simulator output in model space.

        The simulator exists in the problem space, while in the acquisition function we work in the model space. We need
        to correctly transform the x point into problem space and transform the result back into the model space.
        The model space is specific to the GP being worked with.

        Args:
            x (n,d): x points to process (in the "model" space)

        Returns:
            x (n,d): x points to process (in the "model" space)
            y (n,m): the simulator response (in the "model" space)
            yvar (n,m): yvar (observation noise) returned by the simulator (in the "model" space)

        Notes:
            - the `n` used here corresponds to 't_batch` in the forward method.
            - This doesn't not support batches as we use are not expecting multiple y observations.
        """
        # Transform into problem space
        x_prb = self.input_transform.untransform(x)

        # do this to make sure not fitting params.
        # TODO(sw 2025-04-24): this is a general risk with using transforms, how big a problem is it?
        _ = self.outcome_transform.eval()
        # TODO(sw 2025-04-24): would be nicer if we can fix the evaluation function to work with multiple dim.
        y_models = []
        y_var_models = []
        for point in x_prb.numpy():
            sim_result = self.evaluation_function.evaluate(point)

            # TODO(sw 2025-04-24): This ignore covariance between results. Currently we use an independent GP for each
            # target but this will need to be updated if correlation is considered
            y_prb = torch.tensor(sim_result.means)
            y_var_prb = torch.tensor(sim_result.cov).diag()

            # Transform back into model space
            y_model, y_var_model = self.outcome_transform(Y=y_prb, Yvar=y_var_prb)
            y_models.append(y_model)
            y_var_models.append(y_var_model)

        return x, torch.concat(y_models, dim=0), torch.concat(y_var_models, dim=0)


# NOTE: all the setting are copied from QoILookAhead.
# TODO(sw 2025-04-25): Check if can just register the function used by the parent class. `_argparse_qoi_look_ahead`
@optimizer_argparse.register(OptimalQoILookAhead)
def _argparse_optimal_qoi_look_ahead(
    # NOTE: this is a bit of a non-standard implementation because we need to update params in a nested dict. Would
    #  prefer to set the key values directly here
    acqf: OptimalQoILookAhead,
    # needs to accept the variety of args it is handed, and then pick the relevant ones
    **kwargs: Any,  # noqa: ANN401
) -> dict[str, Any]:
    """See axtreme.acquisition.qoi_look_ahead._argparse_qoi_look_ahead docs for details."""
    # Start by using the default arg constructure, then adding in any kwargs that were passed in.

    # NOTE: this is an internal method, using it is an anti pattern.
    # - Ax just stores all the arge parsers in the same file as _argparse_base so they can use it directly.
    # - We can't put this function in that file, even though it belongs there.
    # Definition of _argparse_base explains the shape returned
    args = _argparse_base(acqf, **kwargs)

    # Only update with these defaults if the variable were not passing in kwargs
    optimizer_options = kwargs["optimizer_options"]
    options = optimizer_options.get("options", {})

    if "raw_samples" not in optimizer_options:
        args["raw_samples"] = 100
    if "with_grad" not in options:
        args["options"]["with_grad"] = False

    if "method" not in options:
        args["options"]["method"] = "Nelder-Mead"
        # These options are specific to using Nelder-Mead
        if "maxfev" not in options:
            args["options"]["maxfev"] = "5"
        if "retry_on_optimization_warning" not in optimizer_options:
            args["retry_on_optimization_warning"] = False

    return args


@acqf_input_constructor(OptimalQoILookAhead)
def construct_inputs_optimal_qoi_look_ahead(
    model: SingleTaskGP,
    qoi_estimator: QoIEstimator,
    evaluation_function: EvaluationFunction,
    input_transform: InputTransform,
    outcome_transform: OutcomeTransform,
    **_: dict[str, Any],
) -> dict[str, Any]:
    """See axtreme.acquisition.qoi_look_ahead.construct_inputs_qoi_look_ahead for details."""
    return {
        "model": model,
        "qoi_estimator": qoi_estimator,
        "evaluation_function": evaluation_function,
        "input_transform": input_transform,
        "outcome_transform": outcome_transform,
    }
