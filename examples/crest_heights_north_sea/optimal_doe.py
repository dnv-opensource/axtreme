"""New Acqusition function for Optimal DOE utilizing the experiment directly, as of 2025-05-28 not working."""

# Code for running this acquisition function is not working yet, but it is implemented in the optimal_doe:run.py file.
# Currently trying to run with run_trails resuts in the following error:
# RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
# %%
import copy
from typing import Any

import torch
from ax.core import Arm, Experiment
from ax.modelbridge.registry import Models
from ax.models.torch.botorch_modular.optimizer_argparse import _argparse_base, optimizer_argparse
from botorch.acquisition.input_constructors import acqf_input_constructor
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import InputTransform
from botorch.utils.transforms import t_batch_mode_transform

from axtreme.acquisition.qoi_look_ahead import QoILookAhead
from axtreme.qoi.qoi_estimator import QoIEstimator
from axtreme.utils import transforms


class ExperimentBasedAcquisition(QoILookAhead):
    def __init__(
        self,
        model: SingleTaskGP,
        qoi_estimator: QoIEstimator,
        experiment: Experiment,
        input_transform: InputTransform,
    ) -> None:
        """Args:
        experiment (Experiment): The experiment used to find the brute force optimal with.
        input_transform: The transform that take inputs from problem space to model space.
        """
        super().__init__(model, qoi_estimator)
        self.experiment = experiment
        self.input_transform = input_transform
        self._original_input_transform = qoi_estimator.input_transform
        self._original_outcome_transform = qoi_estimator.outcome_transform

    @t_batch_mode_transform(expected_q=1)
    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Args:
            x (t,q = 1,d): The input to the acquisition function. This is in problem space.

        Returns:
            torch.Tensor: The acquisition function value.
        """
        x = X.squeeze(1)
        # Transform the input to the model
        # the point needsd to be converted from model spact to problem space before it can be used in the experiment.
        x_prb = self.input_transform.untransform(x)
        # Apply _calcuate_point to each input
        results = []
        # for point in x_prb.numpy():
        for point in x_prb:
            y_prb, y_var_prb = self._calulate_point(point)
            results.append(y_var_prb)

        # return -torch.stack(results, dim=0)
        return torch.stack(results, dim=0)

    def _calulate_point(self, point: torch.Tensor) -> torch.Tensor:
        """Args:
            point: one point in the input to the acquisition function. This is needs to be in the Model space.

        Returns:
            torch.Tensor: The acquisition function value.
        """
        # Transform the input to the model

        exp = copy.deepcopy(self.experiment)
        qoi_estimator = copy.deepcopy(self.qoi)
        # qoi_estimator = self.qoi

        trial = exp.new_trial()
        trial.add_arm(Arm(parameters={k: v.item() for k, v in zip(exp.parameters.keys(), point, strict=True)}))
        # trial.add_arm(Arm(parameters={k: v.item() for k, v in zip(exp.parameters.keys(), point[0], strict=True)}))

        _ = trial.run()
        _ = trial.mark_completed()

        # Train a model
        model_bridge = Models.BOTORCH_MODULAR(
            experiment=exp,
            data=exp.fetch_data(),
            # data=exp.fetch_data(metrics=list(exp.optimization_config.metrics.values())),  # type: ignore  # noqa: PGH003
            fit_tracking_metrics=False,  # Needed for QoIMetric to work properly
        )
        input_transform, outcome_transform = transforms.ax_to_botorch_transform_input_output(
            transforms=list(model_bridge.transforms.values()), outcome_names=model_bridge.outcomes
        )

        # Adding the transforms to the QoI estimator
        model = model_bridge.model.surrogate.model
        original_input_transform = qoi_estimator.input_transform
        original_outcome_transform = qoi_estimator.outcome_transform
        # self.qoi.input_transform = input_transform
        # self.qoi.outcome_transform = outcome_transform
        qoi_estimator.input_transform = input_transform
        qoi_estimator.outcome_transform = outcome_transform

        """
        with torch.no_grad():
            # est = QOI_ESTIMATOR(model)
            # est = self.qoi(model)
            est = qoi_estimator(model)
            # est = est.clone().detach()

            # y_prb = self.qoi.mean(est)
            # y_var_prb = self.qoi.var(est)
            y_prb = qoi_estimator.mean(est)
            y_var_prb = qoi_estimator.var(est)

        est = qoi_estimator(model)

        y_prb = qoi_estimator.mean(est)
        y_var_prb = qoi_estimator.var(est)
        """
        with torch.no_grad():
            est = qoi_estimator(model)
            y_prb = qoi_estimator.mean(est)
            y_var_prb = qoi_estimator.var(est)

            # Ensure outputs don't require gradients
            y_prb = y_prb.detach() if y_prb.requires_grad else y_prb
            y_var_prb = y_var_prb.detach() if y_var_prb.requires_grad else y_var_prb

        # Reset the transforms
        # qoi_estimator.input_transform = original_input_transform
        # qoi_estimator.outcome_transform = original_outcome_transform
        qoi_estimator.input_transform = self._original_input_transform
        qoi_estimator.outcome_transform = self._original_outcome_transform

        return y_prb, y_var_prb


# NOTE: all the setting are copied from QoILookAhead.(hsb 2025-05-27)
@optimizer_argparse.register(ExperimentBasedAcquisition)
def _argparse_experiment_based_acquisition(
    # NOTE: this is a bit of a non-standard implementation because we need to update params in a nested dict. Would
    #  prefer to set the key values directly here
    acqf: ExperimentBasedAcquisition,
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


@acqf_input_constructor(ExperimentBasedAcquisition)
def construct_inputs_experiment_based_acquisition(
    model: SingleTaskGP,
    qoi_estimator: QoIEstimator,
    experiment: Experiment,
    input_transform: InputTransform,
    **_: dict[str, Any],
) -> dict[str, Any]:
    """See axtreme.acquisition.qoi_look_ahead.construct_inputs_qoi_look_ahead for details."""
    return {
        "model": model,
        "qoi_estimator": qoi_estimator,
        "experiment": experiment,
        "input_transform": input_transform,
    }
