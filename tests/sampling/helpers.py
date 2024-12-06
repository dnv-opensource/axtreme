"""Mainly provides the different posterior to test sampling against.

Assume covariance (off diagonals) is 0 unless function mentions otherwise

These are defined as functions rather than fixtures as we often refere to them with `@pytest.mark.paramterise`.
Pytest does not allow you to refere to fixtures in parameterise. For discussion of alternative see
`docs/dev/design_decisons/pytest.md`.

NOTE: This uses the posterior dimension convention defined [here](https://docs.gpytorch.ai/en/stable/distributions.html#multitaskmultivariatenormal)
NOTE: This could be within conftest, but appears it is considerd bad practice to import from conftest https://github.com/pytest-dev/pytest/issues/3272
"""

# %%
import torch
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.distributions.multitask_multivariate_normal import MultitaskMultivariateNormal
from gpytorch.distributions.multivariate_normal import MultivariateNormal


def posterior_n1_t1() -> GPyTorchPosterior:
    """Create a posterior (represents gp prediction output).

    Represents:
        - 1 input point
        - 1d output dimension

    Resultant shape:
    >>> posterior.distribution.mean
    tensor([1])
    >>> posterior.distribution.covariance_matrix
    tensor([[1.]])

    Why test this case:
        - most basic example of a prediction
    """
    return GPyTorchPosterior(MultivariateNormal(mean=torch.tensor([1.0]), covariance_matrix=torch.eye(1)))


def posterior_n2_t1() -> GPyTorchPosterior:
    """Create a posterior (represents gp prediction output).

    Represents:
        - 2 input point.
        - 1d output dimension.
        - No covariance.

    Resultant shape:
    >>> posterior.distribution.mean
    tensor([1,1])
    >>> posterior.distribution.covariance_matrix
    tensor([[1., 0.],
            [0., 1.]]))

    Why test this case:
        - Show how shape extends with additional inputs compared to `posterior_1point_1Doutput`
    """
    return GPyTorchPosterior(MultivariateNormal(mean=torch.tensor([1.0, 1]), covariance_matrix=torch.eye(2)))


def posterior_n1_t2() -> GPyTorchPosterior:
    """Create a posterior (represents gp prediction output).

    Represents:
        - 1 input point.
        - 2d output dimension.
        - Independat targets
        - Within targets, independant x points.

    Resultant shape:
    >>> posterior.distribution.mean
    tensor([[1.0000, 0.5000]])
    >>> posterior.distribution.covariance_matrix
    tensor([[1.0000, 0.0000],
            [0.0000, 0.5000]]

    Why test this case:
        - Basic example of multi dimensional output space
    """
    mvn_task1 = MultivariateNormal(mean=torch.tensor([1.0]), covariance_matrix=torch.eye(1))
    mvn_task2 = MultivariateNormal(mean=torch.tensor([0.5]), covariance_matrix=torch.eye(1) * 0.5)
    return GPyTorchPosterior(MultitaskMultivariateNormal.from_independent_mvns([mvn_task1, mvn_task2]))


def posterior_n3_t2() -> GPyTorchPosterior:
    """Create a posterior (represents gp prediction output).

    Represents:
        - 3 input point.
        - 2d output dimension.
        - Independat targets
        - Within targets, independant x points.

    Resultant shape:
    >>> posterior.distribution.mean
    tensor([[1.0000, 0.5000],
            [1.0000, 0.5000],
            [1.0000, 0.5000]])
    >>> posterior.distribution.covariance_matrix
    tensor([[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
            [0.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000],
            [0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.0000, 0.5000, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.0000, 0.0000, 0.5000, 0.0000],
            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.5000]]))

    Why test this case:
        - Show how shape extends with additional inputs compared to `posterior_1point_2Doutput`
    """
    mvn_task1 = MultivariateNormal(mean=torch.tensor([1.0, 1, 1]), covariance_matrix=torch.eye(3))
    mvn_task2 = MultivariateNormal(mean=torch.tensor([0.5, 5, 5]), covariance_matrix=torch.eye(3) * 0.5)
    return GPyTorchPosterior(MultitaskMultivariateNormal.from_independent_mvns([mvn_task1, mvn_task2]))


def posterior_n3_t2_interleaved() -> GPyTorchPosterior:
    """Creates a Multitask Multivariate Normal with interleaved tasks.

    In interleaved format the covariance matrix groups all targets for the same x point together. The mean is always
    (b,n,m). See example below.

        Non-interleaved format:
            x_1, ..., x_n, x_1, ..., x_n,
        x_1|              |               |
        ...|   Task 1     |   Cov Task1   |
        x_n|              |   & Task 2    |
        - |--------------|---------------|
        x_1|              |               |
        ...|   Cov Task1  |     Task 2    |
        x_n|   & Task 2   |               |
        --------------------------------

        Interleaved format:
            x_1, x1,..., ..., x_n, x_n,
        x_1| t1,
        x_1|   , t2
        ...|
        ...|
        x_n|
        x_n|

    Example:

        non_interleaved_cov = torch.tensor([
            [1.0,  0.0,  0.0,  0.0,  0.0,  0.0],
            [0.0,  2.0,  0.0,  0.0,  0.0,  0.0],
            [0.0,  0.0,  3.0,  0.0,  0.0,  0.0],
            [0.0,  0.0,  0.0, 11.0,  0.0,  0.0],
            [0.0,  0.0,  0.0,  0.0, 12.0,  0.0],
            [0.0,  0.0,  0.0,  0.0,  0.0, 13.0],
        ])

        interleaved_cov = torch.tensor([
            [1.0, 0.0,  0.0,  0.0,  0.0,  0.0],
            [0.0, 11.0, 0.0,  0.0,  0.0,  0.0],
            [0.0,  0.0, 2.0,  0.0,  0.0,  0.0],
            [0.0,  0.0, 0.0, 12.0,  0.0,  0.0],
            [0.0,  0.0, 0.0,  0.0,  3.0,  0.0],
            [0.0,  0.0, 0.0,  0.0,  0.0, 13.0],
        ],


    Why test this case:
        - MultitaskMultivariateNormal: Support both formats. Should ensure both are handled.
    """

    # fmt: off
    # Shape (*b,n,m)
    mean = torch.tensor(
        [
            [1 ,100],
            [2, 200],
            [3, 300],
        ]
    )

    cov_interleaved = torch.tensor(
        [
            [1.0, 0.0,  0.0,  0.0,  0.0,  0.0],
            [0.0, 11.0, 0.0,  0.0,  0.0,  0.0],
            [0.0,  0.0, 2.0,  0.0,  0.0,  0.0],
            [0.0,  0.0, 0.0, 12.0,  0.0,  0.0],
            [0.0,  0.0, 0.0,  0.0,  3.0,  0.0],
            [0.0,  0.0, 0.0,  0.0,  0.0, 13.0],
        ],
    )
    # fmt: on

    return GPyTorchPosterior(
        MultitaskMultivariateNormal(mean=mean, covariance_matrix=cov_interleaved, interleaved=True)
    )


def posterior_b2_n3_t2() -> GPyTorchPosterior:
    """Create a posterior (represents gp prediction output).

    Represents:
        - 2 batches
        - 3 input point.
        - 2d output dimension.
        - Independat targets
        - Within targets, independant x points.

    Resultant shape:
    >>> posterior.distribution.mean.shape
    torch.Size([2,3,2])
    >>> posterior.distribution.covariance_matrix.shape
    torch.Size([2,3*2,3*2])


    Why test this case:
        - Add posterior batch shape to `posterior_3point_2Doutput`.
    """
    # fmt: off
    batch_mean = torch.tensor(
        [[[1., 1.5],
          [1., 1.5],
          [1., 1.5]],
         [[2., 2.5],
          [2., 2.5],
          [2., 2.5]]]
    )
    # fmt: on

    batch_covariance = torch.tensor(
        [
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.5, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.5, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.5],
            ],
            [
                [2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 2.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 2.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 2.5, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 2.5, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 2.5],
            ],
        ]
    )

    return GPyTorchPosterior(
        MultitaskMultivariateNormal(mean=batch_mean, covariance_matrix=batch_covariance, interleaved=False)
    )


# %%
if __name__ == "__main__":
    """The following can be used to generate more posteriors should they be needed."""

    import torch
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import SingleTaskGP
    from gpytorch.mlls import ExactMarginalLogLikelihood

    torch.set_default_dtype(torch.float64)

    def standardize_data(data: torch.Tensor):
        mean = data.mean(dim=0)
        std = data.std(dim=0)
        return (data - mean) / std

    def max_min_normalize_data(data: torch.Tensor):
        max_val = data.max(dim=0).values  # noqa: PD011
        min_val = data.min(dim=0).values  # noqa: PD011
        return (data - min_val) / (max_val - min_val)

    def make_posterior(n_test_points: int, y_dim: int, n_training_points: int = 10):
        # x dim is not included as it doesn't not effect the posterior dim
        train_x = max_min_normalize_data(torch.rand([n_training_points, 1]))
        train_y = standardize_data(torch.rand(n_training_points, y_dim))

        model = SingleTaskGP(train_x, train_y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        _ = fit_gpytorch_mll(mll)
        test_x = torch.rand([n_test_points, 1])
        with torch.no_grad():
            posterior = model.posterior(test_x)

        return posterior

    # posterior = make_posterior(n_test_points=2, y_dim=2)
    n_test_points = 3
    y_dim = 2
    n_training_points = 10
    # x dim is not included as it doesn't not effect the posterior dim
    train_x = max_min_normalize_data(torch.rand([2, n_training_points, 1]))
    train_y = standardize_data(torch.rand(2, n_training_points, y_dim))

    model = SingleTaskGP(train_x, train_y)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    _ = fit_gpytorch_mll(mll)
    test_x = torch.rand([n_test_points, 1])
    with torch.no_grad():
        posterior = model.posterior(test_x)

    assert isinstance(posterior, GPyTorchPosterior)

    dist = posterior.distribution
    print(f"{dist.mean.shape= }, {dist.covariance_matrix.shape= }")  # noqa: T201
    # print(f"{type(dist)}\n{dist.mean}\n{dist.covariance_matrix}")

    # %%
    dist = posterior_b2_n3_t2()
    print(f"{dist.mean.shape= }, {dist.covariance_matrix.shape= }")  # noqa: T201

    # %%
    dist.shape()
    # %%
    dist  # noqa: B018  # pyright: ignore[reportUnusedExpression]
    # %%
