"Util funciton for working with QoIEstimators."

import copy
from typing import TypeVar

from ax.modelbridge import ModelBridge
from git import TYPE_CHECKING

from axtreme.utils import transforms

if TYPE_CHECKING:
    from axtreme.qoi import QoIEstimator


T = TypeVar("T", bound="QoIEstimator")


def attach_transforms_to_qoi_estimator(model_bridge: ModelBridge, qoi_estimator: T) -> T:
    """Attaches appropriate botorch transforms to a QoIEstimator.

    NOTE: this approach is a hick fix to a wider problem we are yet to solve elegantly. Our QoIEstimators need to
    opeate in the "problem space" but the botorch models passed to the QoIEstimator have been put in the "Model space"
    by ax.

    The current approach is to find the botorch transforms that should be applied, and give them to the QoIEstimator.
    This allows QoIEstimate in the problem space without interfering with the ax pattern of the botorch model being in
    the "model space".

    TODO(sw 14-4-25): This approach has proven clunky to work with, and the attributes are not documented on
    `QoIEstimator`. Explore alternative (issues #19).
    """
    qoi_estimator = copy.deepcopy(qoi_estimator)
    input_transform, outcome_transform = transforms.ax_to_botorch_transform_input_output(
        transforms=list(model_bridge.transforms.values()),
        outcome_names=model_bridge.outcomes,
    )

    qoi_estimator.input_transform = input_transform  # type: ignore[attr-defined]
    qoi_estimator.outcome_transform = outcome_transform  # type: ignore[attr-defined]

    return qoi_estimator
