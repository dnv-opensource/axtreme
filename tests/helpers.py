"""A variety of different types of models.

Key attributes of the model change dimension based on:
- the number of outputs (m)
- the type of noise

It is invariant to:
- (n): the number of input datapoints
- (d): the dimensionality of input datapoints

NOTE: these are not fixture as then you can not use them in parameterise
NOTE: This could be within conftest, but appears it is considerd bad practice to import from conftest https://github.com/pytest-dev/pytest/issues/3272
"""

import torch
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms.outcome import Standardize

torch.set_default_dtype(torch.float64)


def get_train_x() -> torch.Tensor:
    """X dimensions are not effected by other setting, so one is used for all veriations."""
    gen = torch.Generator()
    _ = gen.manual_seed(7)
    return torch.rand(10, 1, dtype=torch.float64, generator=gen)


def single_task_homo_noise_m_1():
    """Produce the following shapes for its attributes.

    NOTE: hyperparameters of this model have not been fit

    `model.train_inputs`: ((n,d),)
    `model.train_targets`: (n,)
    `model.likelihood.noise_covar.noise`: (1,)
    """
    # (n,1)
    train_x = get_train_x()
    # (n,1)
    train_y = torch.sin(train_x)
    # Stops it complaining about non-centred input
    outcome_transform = Standardize(m=1)
    std_y = outcome_transform(train_y)[0]
    return SingleTaskGP(train_x, std_y)


def single_task_homo_noise_m_2():
    """Produce the following shapes for its attributes.

    NOTE: hyperparameters of this model have not been fit

    `model.train_inputs`: ((m,n,d),)
    `model.train_targets`: (m,n)
    `model.likelihood.noise_covar.noise`: (m,1)
    """
    # (n,1)
    train_x = get_train_x()  # (n,1)
    # (n,m = 2)
    train_y = torch.concat([torch.sin(train_x), torch.cos(train_x)], dim=-1)
    # Stops it complaining about non-centred input
    outcome_transform = Standardize(m=2)
    std_y = outcome_transform(train_y)[0]
    return SingleTaskGP(train_x, std_y)


def single_task_fixed_noise_m_1():
    """Produce the following shapes for its attributes.

    NOTE: hyperparameters of this model have not been fit

    `model.train_inputs`: ((n,d),)
    `model.train_targets`: (n,)
    `model.likelihood.noise_covar.noise`: (n,)
    """
    # (n,1)
    train_x = get_train_x()
    # (n,1)
    train_y = torch.sin(train_x)
    train_yvar = torch.linspace(0.01, 0.2, train_y.numel()).reshape_as(train_y)
    # Stops it complaining about non-centred input
    outcome_transform = Standardize(m=1)
    std_y, std_var = outcome_transform(train_y, train_yvar)
    return SingleTaskGP(train_x, std_y, std_var)


def single_task_fixed_noise_m_2():
    """Produce the following shapes for its attributes.

    NOTE: hyperparameters of this model have not been fit

    `model.train_inputs`: ((m,n,d),)
    `model.train_targets`: (m,n)
    `model.likelihood.noise_covar.noise`: (m,n)
    """
    # (n,1)
    train_x = get_train_x()
    # (n,m = 2)
    train_y = torch.concat([torch.sin(train_x), torch.cos(train_x)], dim=-1)
    train_yvar = torch.linspace(0.01, 0.2, train_y.numel()).reshape_as(train_y)
    # Stops it complaining about non-centred input
    outcome_transform = Standardize(m=2)
    std_y, std_var = outcome_transform(train_y, train_yvar)
    return SingleTaskGP(train_x, std_y, std_var)


def single_task_homo_noise_m_2_b3():
    """Produce the following shapes for its attributes.

    NOTE: hyperparameters of this model have not been fit.

    `model.train_inputs`: ((b,m,n,d),)
    `model.train_targets`: (b,m,n)
    `model.likelihood.noise_covar.noise`: (b,m,1)
    """
    # (n,1)
    train_x = get_train_x()  # (n,1)
    train_x = train_x.expand(3, *train_x.shape)
    # (b =3, n = 10,m = 2)
    train_y = torch.concat([torch.sin(train_x), torch.cos(train_x)], dim=-1)
    # Stops it complaining about non-centred input
    outcome_transform = Standardize(m=2, batch_shape=torch.Size([3]))
    std_y = outcome_transform(train_y)[0]
    return SingleTaskGP(train_x, std_y)


def single_task_homo_noise_m_1_b3():
    """Produce the following shapes for its attributes.

    NOTE: hyperparameters of this model have not been fit.

    `model.train_inputs`: ((b,n,d),)
    `model.train_targets`: (b,n)
    `model.likelihood.noise_covar.noise`: (b,1)
    """
    # (n,1)
    train_x = get_train_x()  # (n,1)
    train_x = train_x.expand(3, *train_x.shape)
    # (b =3, n = 10,m = 1)
    train_y = torch.sin(train_x)
    # Stops it complaining about non-centred input
    outcome_transform = Standardize(m=1, batch_shape=torch.Size([3]))
    std_y = outcome_transform(train_y)[0]
    return SingleTaskGP(train_x, std_y)


def single_task_fixed_noise_m_1_outcome_transform():
    """Produce the following shapes for its attributes.

    NOTE: hyperparameters of this model have not been fit

    `model.train_inputs`: ((n,d),)
    `model.train_targets`: (n,)
    `model.likelihood.noise_covar.noise`: (n,)
    """
    # (n,1)
    train_x = get_train_x()
    # (n,1)
    train_y = torch.sin(train_x)
    train_yvar = torch.linspace(0.01, 0.2, train_y.numel()).reshape_as(train_y)
    return SingleTaskGP(train_x, train_y, train_yvar, outcome_transform=Standardize(m=1))
