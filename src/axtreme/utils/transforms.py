"""The module is used to determine the ax transformations used and create the equivalent botorch transformation.

This allows the user to work directly with the Botorch model, while using inputs and outputs in the original space
(e.g the problem space)

Todo:
    It would be nice to be able to return an identity transform so we don't have to deal with Nones (sw 2024-11-21).
"""

from collections.abc import Callable
from functools import partial
from typing import Any, TypeAlias

import torch
from ax.modelbridge.transforms.base import Transform
from ax.modelbridge.transforms.cast import Cast
from ax.modelbridge.transforms.choice_encode import OrderedChoiceEncode
from ax.modelbridge.transforms.derelativize import Derelativize
from ax.modelbridge.transforms.int_to_float import IntToFloat
from ax.modelbridge.transforms.ivw import IVW
from ax.modelbridge.transforms.log import Log
from ax.modelbridge.transforms.logit import Logit
from ax.modelbridge.transforms.one_hot import OneHot
from ax.modelbridge.transforms.remove_fixed import RemoveFixed
from ax.modelbridge.transforms.standardize_y import StandardizeY
from ax.modelbridge.transforms.unit_x import UnitX
from botorch.models.transforms.input import InputTransform, Normalize
from botorch.models.transforms.outcome import OutcomeTransform, Standardize

TransformationTranslation: TypeAlias = Callable[
    [Any],  # NOTE: This should be `Transform`, but making types work in ax_to_botorch_transform_conversion is tricky.
    tuple[
        InputTransform | None,
        OutcomeTransform | None,
    ],
]


def ax_to_botorch_transform_input_output(
    transforms: list[Transform], outcome_names: list[str]
) -> tuple[InputTransform, OutcomeTransform]:
    """Determines the input and output transforms applied by Ax, and creates the equivalent transforms in botroch.

    This allows the botorch model internal to ax (which operates in a standard "model" space), to be used in the problem
    space (e.g non-standardised input and output). This is useful when calculating QoIs.

    Args:
        transforms: the TRAINED transforms that have been applied by `ax`.
            - Typically found at `TorchModelBridge.transforms`.
        outcome_names: the order of the output columns used to train the internal ax.Model.
            - Typically found at `TorchModelBridge.outcomes`.

    Return:
        - input_transform:
        - output_tranform:

    Using them in the following way allow input and output in the outcome/problem space:
        - Assume: `model` is a trained`botorch.models`(such as `TorchModelBridge.model.surrogate.model`)
        - > model.posterior(input_transform(X), posterior_transform = output_transform.untransform_posterior)

    Todo:
        - Ideally `ax_to_botorch_transform_conversion` would a config within the root of this module, so it could easily
            be exteneded. This is challenign becuase the 'translate_standardisey' function needs the specific
            outcome_names of the problem to be passed. This is because ax does not maintain the order of the metrics in
            the transform itself (it stores the names/order internally. See
            `ax.modelbridge.base.ModelBridge._transform_data` for details)
    """
    # This is a mapping from the ax Tranform to the function that translates it to botorch
    # Key: the specific transform class,
    # Value: the function to extract the input and outcome transforms
    #
    # NOTE: The transforms that are listed here are the standard transforms applied to model.BOTORCH_MODULAR.
    # They are detailed here:  ax.modelbridge.registry.MODEL_KEY_TO_MODEL_SETUP
    ax_to_botorch_transform_conversion: dict[type[Transform], TransformationTranslation] = {
        Cast: translate_cast,
        RemoveFixed: partial(check_transform_not_applied, parameter_names_store="fixed_parameters"),
        OrderedChoiceEncode: partial(check_transform_not_applied, parameter_names_store="encoded_parameters"),
        OneHot: partial(check_transform_not_applied, parameter_names_store="encoded_parameters"),
        IntToFloat: partial(check_transform_not_applied, parameter_names_store="transform_parameters"),
        Log: partial(check_transform_not_applied, parameter_names_store="transform_parameters"),
        Logit: partial(check_transform_not_applied, parameter_names_store="transform_parameters"),
        UnitX: translate_unitx,
        IVW: translate_ivw,
        Derelativize: translate_derelativize,
        # TODO(sw): Would be great to work out how to avoid passing col_names
        # Ax stores the order it uses here: `TorchModelBridge.outcomes`
        StandardizeY: partial(translate_standardisey, col_names=outcome_names),
    }

    input_transforms: list[InputTransform] = []
    output_transforms: list[OutcomeTransform] = []

    for t in transforms:
        try:
            converter = ax_to_botorch_transform_conversion[t.__class__]
        except KeyError:
            msg = f"ax_to_botorch_transform_conversion does not support {t.__class__}"
            raise NotImplementedError(msg) from None

        input_transform: InputTransform | None
        output_transform: OutcomeTransform | None
        input_transform, output_transform = converter(t)

        if input_transform is not None:
            input_transforms.append(input_transform)
        if output_transform is not None:
            output_transforms.append(output_transform)

    # TODO(sw 2024-11-21): Look at extending support to multiple transforms using:
    # botorch.models.transforms.outcome.ChainedOutcomeTransform
    # botorch.models.transforms.input.ChainedInputTransform
    if len(input_transforms) > 1:
        raise NotImplementedError(f"Currently only a single input transform is supported, found {input_transforms}")
    if len(output_transforms) > 1:
        raise NotImplementedError(f"Currently only a single output transform is supported, found {output_transforms}")

    return input_transforms[0], output_transforms[0]


#### Individual translation function


# TODO(sw): Maybe this is cleaner as an object rather than using functools?
def check_transform_not_applied(
    transform: Transform,
    parameter_names_store: str,
) -> tuple[InputTransform | None, OutcomeTransform | None]:
    """Used to ensure a transform has not been applied.

    Many transforms store an internal list of the ax.parameters (input) they should be applied to.
    This is determined by that parameter being of a specific type and having specific attributes as checked within the
    transform.
    This helper function is used to double check the transforms are not being used/applied to anything. This mean a
    translation from ax to botorch is not required.

    Args:
        transform: the transform to check
        parameter_names_store: The attribute on the transform that should be empty (falsey) if the tranform is not
            applied to anything.

    Returns:
        Input and output transforms required (will be None). Will raise an error if these transformation have actually
        been applied.

    Note:
        This should be instantiated with functools.partial, e.g.
            >>> from functools import partial
            >>> log_checker = partial(check_not_applied, parameter_names_store="transform_parameters")

    """
    # There are no names stored
    if not bool(getattr(transform, parameter_names_store)):
        return (None, None)

    msg = (
        f"expected {transform.__class__}.{parameter_names_store} to be empty/falsey,"
        f" instead found {getattr(transform, parameter_names_store)} indicating this transform is being used"
    )
    raise AssertionError(msg)


def translate_cast(transform: Cast) -> tuple[InputTransform | None, OutcomeTransform | None]:
    """Make sure that Cast has not flattned a HierachicalSearchSpace.

    Cast changes the parameter (e.g RangeParameter), castings the VALUE to the data type it should be.
        e.g RangeParameter values should be a float, cast the value to ensure it is a float
    It also deals with HierachicalSearchSpace:
        - (basically this is like a tree that navigates you to a more specific search space
            - e.g if 'parameter_a'> 2 -> use SearchSpace1
        - `.flatten_hss` flag if this has been used
    """
    if transform.flatten_hss:
        msg = "Ax transfrom to Botorch transform for Cast does not currently support HierachicalSearchSpace"
        raise NotImplementedError(msg)

    # No transform as it is assumed the inputs used for the botorch model are of they right type (e.g torch.tensor)
    return (None, None)


def translate_unitx(transform: UnitX) -> tuple[InputTransform | None, OutcomeTransform | None]:
    """Converts a trained UnitX to botorch equivalent.

    Ax bounds look like this: {'x1': (0.0, 2.0), 'x2': (0.0, 3.0)}

    BoTorch bounds look like: tensor([[0., 0.],[2., 3.]])
    """
    bounds = transform.bounds
    normalise = Normalize(d=len(bounds), bounds=torch.tensor(list(bounds.values())).T)
    _ = normalise.eval()
    return (normalise, None)


def translate_ivw(transform: IVW) -> tuple[InputTransform | None, OutcomeTransform | None]:  # noqa: ARG001
    """Handle IVW (inverse variance weight transform).

    IVW is used when at the same location (x), there are multiple measure of the same metric.
    It combines these into a single measure of the metric, and passes this on to the botroch model for training
    As this is only using for training (transforming the y input to the model), we can ignore this, as we currently
    use these transforms for prediction only.

    Note:
        It is hard to tell if this transformation has been applied because no attribute are stored on the object.

    Todo:
        Check if botorch supports multiple measure of a single metric at a single point (suspect not)
        - if not it is reasonable to ignore this transformation as standard botorch model shouldn't be using in that way
    """
    return (None, None)


def translate_derelativize(transform: Derelativize) -> tuple[InputTransform | None, OutcomeTransform | None]:  # noqa: ARG001
    """Handle Derelativize (relative constraints to non-relative constraints).

    Derelativize transforms optimisation configs and untransforms constraints. As we are only interested in input and
    output transformations, this can be ignored.

    Todo:
        Is there a safer way to ensure this is not being used? Difficult because it doesn't store anything internally

    Todo:
        This needs some additional work.
    """
    return (None, None)


def translate_standardisey(
    transform: StandardizeY,
    col_names: list[str],
) -> tuple[InputTransform | None, OutcomeTransform | None]:
    """Translate ax standardisation into botorch standardisation.

    Note:
        Ax does not maintain the order of the metrics, so need to explicitly pass the order.

        .. Todo::
            Can there be some work around for this? Would be good not to have to pass constraints.

    Note:
        col_name should be passed using functools.partial
            >>> from functools import partial
            >>> standardise_y = partial(translate_standardiseY, col_names=["loc", "scale"])

    Args:
        transform: StandardizeY to translate
        col_names: the order of the column in the data being passed in.
            This is required to the correct transformation can be applied to the correct column
    """
    # TODO(sw): this currently assumes all outputs are being standardises
    assert set(col_names) == set(transform.Ymean.keys())

    new_output_transform = Standardize(m=len(transform.Ymean))
    new_output_transform.means = torch.tensor([[transform.Ymean[name] for name in col_names]])
    new_output_transform.stdvs = torch.tensor([[transform.Ystd[name] for name in col_names]])
    # Need to set this directly because it is used to give the covariance
    new_output_transform._stdvs_sq = new_output_transform.stdvs.pow(2)  # noqa: SLF001
    new_output_transform._is_trained = torch.tensor([True])  # noqa: SLF001
    _ = new_output_transform.eval()

    return (None, new_output_transform)
