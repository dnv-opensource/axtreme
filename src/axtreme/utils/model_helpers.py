"""Utility functions for working with botorch models."""

import torch
from botorch.models import SingleTaskGP
from gpytorch.likelihoods.gaussian_likelihood import FixedNoiseGaussianLikelihood, GaussianLikelihood


def get_training_data_singletaskgp(model: SingleTaskGP) -> torch.Tensor:
    """Extracts the training data from SIngleTaskGP in consistent form.

    As per Botorch documentation, "SingleTaskGP Modules same training data for all outputs" (`BoTorch docs <https://botorch.org/docs/models#single-task-gps>`_)
    As such we are not concerned with batch dimenstion. Gpytorch stores training data in this format:

        - if m = 1: (`*b`,n,d)
        - if m > 1: (`*b`,m,n,d)

    Args:
        model: the model to collect the training data from.

    Returns:
        training_inputs: (n,d) A torch.Tensor of the training_inputs

    NOTE: The interface defined here is only available for SingleTaskGPs, as other model can contain different input
        data for each batch.
    """
    input_data_tuple = model.train_inputs
    assert input_data_tuple is not None, f"model.train_inputs is None: {input_data_tuple}"
    assert len(input_data_tuple) > 0, f"model.train_inputs is empty: {input_data_tuple}"
    if len(input_data_tuple) > 1:
        msg = f"model.train_inputs has more than one element: {input_data_tuple}"
        raise ValueError(msg)

    # Get to the data
    batched_input_data = input_data_tuple[0]

    batch_shape = batched_input_data.shape[:-2]
    index = [0] * len(batch_shape) + [slice(None), slice(None)]
    input_data = batched_input_data[index]
    return input_data


def get_target_noise_singletaskgp(model: SingleTaskGP) -> torch.Tensor:
    """A helper to extract the training varaince from a botorch model, and return it in a consistent format.

    Args:
        model: A SingleTaskGP model.

    Returns:
        noise (variance): A torch.Tensor of the noise with shape `(*b, n,m)` where b is an optional batch dimention, `n`
        is the number of observations, `m` is the number of targets. If homoskedatic noise is used shape will be
        `(*b, n=1,m)`.

    NOTE: Approach is based on FantasizeMixin.fantasize.
    NOTE: Currently unclear if this interface extends to othe type of model.
    """
    # Fixed and Homoskedatic noise
    if isinstance(model.likelihood, FixedNoiseGaussianLikelihood | GaussianLikelihood):
        # this chunk in based on FantasizeMixin.fantasize
        if model.num_outputs > 1:
            # change from (*b, m, n) to (*b,n,m)
            # fixed noise: (*b, m, n) to (*b,n,m)
            # homoskedatic noise: (*b, m, n = 1) to (*b,n = 1,m)
            observation_noise = model.likelihood.noise.transpose(-1, -2)
        else:
            # Change from (*b,n) to (*b,n,m=1)
            # fixed noise: (*b,n = 1) to (*b,n = 1,m=1)
            # homoskedatic noise: (*b,n = 1) to (*b,n = 1,m=1)
            observation_noise = model.likelihood.noise.unsqueeze(-1)

        return observation_noise

    msg = (
        "Expected model.likilihood of `FixedNoiseGaussianLikelihood` or `GaussianLikelihood`,"
        f" got {type(model.likelihood.noise)}"
    )
    raise NotImplementedError(msg)
