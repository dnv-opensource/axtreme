import pytest
import torch
from botorch.models import SingleTaskGP

from tests.acquisition.helpers import check_posterior, check_same_model_data


def get_train_x():
    gen = torch.Generator()
    _ = gen.manual_seed(7)
    return torch.rand(10, 1, dtype=torch.float64, generator=gen)


# lets just write wht whole test without importing any of our stuff for now


# This warning is due to train_y not being standardised. This is not imporant for this test
@pytest.mark.filterwarnings("ignore: Input data is not standardized")
@pytest.mark.external
@pytest.mark.parametrize(
    "n_targets, fixed_noise",
    [
        pytest.param(1, False, id="Single target, homoskedatic noise"),
        pytest.param(2, False, id="Multiple target, homoskedatic noise"),
        pytest.param(1, True, id="Single target, Fixed noise"),
        pytest.param(2, True, id="Multiple target, Fixed noise"),
    ],
)
def test_condition_on_observations(n_targets: int, fixed_noise: bool):  # noqa: FBT001
    """Checks if the conditional/fantasy produces the same model as training with all the data.

    Args:
        n_targets: The number of targets the GP should predict.
        fixed_noise: If True, use fixed noise GP, other wise use infered homoskedastic noise.

    Note: This is possible because `SingleTaskGP` does not fit the hyper parmas, that is done using
    >>> mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
    >>> fit_gpytorch_model(mll)

    Benefits of using condition_on_observations:
    - Input have consistent shape, and match instationation of SingleTaskGp.

    Gotchas:
    - some of the arguments are expected to be in model space, others in output (problem) space. Details https://github.com/pytorch/botorch/issues/2533
    """
    fixed_noise = True
    n_targets = 1

    train_x = get_train_x()
    n = train_x.shape[0]
    train_y = torch.sin(train_x).expand(n, n_targets)
    train_yvar = torch.ones_like(train_y) * 0.1

    model_original = SingleTaskGP(train_x, train_y, train_yvar if fixed_noise else None)

    # run the model once to avoid runtime error detailing this requirement
    _ = model_original.posterior(torch.tensor([[0.5]]))

    # Generate new training data to use
    new_x = torch.tensor([[0.5]])  # (n,d)
    new_y = torch.tensor([[0.1]]).expand(1, n_targets)  # (n, m)
    new_yvar = torch.ones_like(new_y) * 0.1

    # New conditioned model
    model_conditioned = model_original.condition_on_observations(
        # MUST BE IN MODEL SPACE - will not be automatically transformed
        # shape require (b,n',d)
        X=new_x,
        # Must be in outcome space - will be automatically transformed
        # Shape require (b,n',m)
        Y=new_y,
        # Must be in model space - - will not be automatically transformed
        noise=new_yvar if fixed_noise else None,
    )

    # Expected model
    x_total = torch.vstack([train_x, new_x])  # (n+1,d=1)
    y_total = torch.vstack([train_y, new_y])  # (n+1,n_targets)
    yvar_total = torch.vstack([train_yvar, new_yvar])  # (n+1,n_targets)
    model_expected = SingleTaskGP(x_total, y_total, yvar_total if fixed_noise else None)

    # Check the internal data is the same
    check_same_model_data(model_actual=model_conditioned, model_expected=model_expected)
    # check that they both make the same preditiontion
    x = torch.tensor([[0.4], [0.7]])
    expected_posterior = model_expected.posterior(x)
    conditioned_posterior = model_conditioned.posterior(x)
    check_posterior(conditioned_posterior, expected_posterior)


@pytest.mark.external
@pytest.mark.filterwarnings("ignore: Input data is not standardized")
@pytest.mark.xfail(
    reason="Batched GP has been created from non-batch GP",
)
def test_condition_on_observations_additional_data_incorrect_dim():
    """Checks the outcome of the model if adding batched data to a non-batched model.

    OUTCOME: new model hase shape torch.Size([1, 11, 1]), rather than the expected torch.Size([11, 1]).
    This because the model has now been turned into a batched GP
    """
    train_x = get_train_x()
    train_y = torch.sin(train_x)
    train_yvar = torch.ones_like(train_y) * 0.1

    model_original = SingleTaskGP(train_x, train_y, train_yvar)

    # run the model once to avoid runtime error detailing this requirement
    _ = model_original.posterior(torch.tensor([[0.5]]))

    # Make the new data have a batch dimension. If the original model was trained with data of this dimension it would
    # be a batched model (see here: https://botorch.org/docs/batching#batched-evaluation-of-models-and-acquisition-functions)
    new_x = torch.tensor([[[0.5]]])  # (b,n,d)
    new_y = torch.tensor([[[0.1]]])  # (b,n, m)
    new_yvar = torch.ones_like(new_y) * 0.1

    # New conditioned model
    model_conditioned = model_original.condition_on_observations(
        # MUST BE IN MODEL SPACE - will not be automatically transformed
        # shape require (b,n',d)
        X=new_x,
        # Must be in outcome space - will be automatically transformed
        # Shape require (b,n',m)
        Y=new_y,
        # Must be in model space - - will not be automatically transformed
        noise=new_yvar,
    )

    # Expected model
    x_total = torch.vstack([train_x, new_x[0]])  # (n+1,d=1)
    y_total = torch.vstack([train_y, new_y[0]])  # (n+1,n_targets)
    yvar_total = torch.vstack([train_yvar, new_yvar[0]])  # (n+1,n_targets)
    model_expected = SingleTaskGP(x_total, y_total, yvar_total)

    # Check the internal data is the same
    check_same_model_data(model_actual=model_conditioned, model_expected=model_expected)
    # check that they both make the same preditiontion
    x = torch.tensor([[0.4], [0.7]])
    expected_posterior = model_expected.posterior(x)
    conditioned_posterior = model_conditioned.posterior(x)
    check_posterior(conditioned_posterior, expected_posterior)


@pytest.mark.external
@pytest.mark.filterwarnings("ignore: Input data is not standardized")
@pytest.mark.xfail(
    reason="Supplying noise in to update homoskedatic has side effects",
)
def test_condition_on_observations_silent_failure():
    """Demonstate the silent failure when providing noise to a GP that doesn't use it.

    The posteriors produces are not the same. This is an edge case of `test_condition_on_observations`.
    """
    fixed_noise = False
    n_targets = 1

    train_x = get_train_x()
    n = train_x.shape[0]
    train_y = torch.sin(train_x).expand(n, n_targets)
    train_yvar = torch.ones_like(train_y) * 0.1

    model_original = SingleTaskGP(train_x, train_y, train_yvar if fixed_noise else None)

    # run the model once to avoid runtime error detailing this requirement
    _ = model_original.posterior(torch.tensor([[0.5]]))

    # Generate new training data to use
    new_x = torch.tensor([[0.5]])  # (n,d)
    new_y = torch.tensor([[0.1]]).expand(1, n_targets)  # (n, m)
    new_yvar = torch.ones_like(new_y) * 0.1

    # New conditioned model
    model_conditioned = model_original.condition_on_observations(
        # MUST BE IN MODEL SPACE - will not be automatically transformed
        # shape require (b,n',d)
        X=new_x,
        # Must be in outcome space - will be automatically transformed
        # Shape require (b,n',m)
        Y=new_y,
        # Must be in model space - - will not be automatically transformed
        noise=new_yvar,
    )

    # Expected model
    x_total = torch.vstack([train_x, new_x])  # (n+1,d=1)
    y_total = torch.vstack([train_y, new_y])  # (n+1,n_targets)
    yvar_total = torch.vstack([train_yvar, new_yvar])  # (n+1,n_targets)
    model_expected = SingleTaskGP(x_total, y_total, yvar_total if fixed_noise else None)

    # Check the internal data is the same
    check_same_model_data(model_actual=model_conditioned, model_expected=model_expected)
    # check that they both make the same preditiontion
    x = torch.tensor([[0.4], [0.7]])
    expected_posterior = model_expected.posterior(x)
    conditioned_posterior = model_conditioned.posterior(x)

    check_posterior(conditioned_posterior, expected_posterior)


# This warning is due to train_y not being standardised. This is not imporant for this test
@pytest.mark.filterwarnings("ignore: Input data is not standardized")
@pytest.mark.external
@pytest.mark.parametrize(
    "n_targets, fixed_noise",
    [
        pytest.param(1, False, id="Single target, homoskedatic noise"),
        pytest.param(2, False, id="Multiple target, homoskedatic noise"),
        pytest.param(1, True, id="Single target, Fixed noise"),
        pytest.param(2, True, id="Multiple target, Fixed noise"),
    ],
)
def test_get_fantasy_model(n_targets: int, fixed_noise: bool):  # noqa: FBT001
    """Checks if the conditional/fantasy produces the same model as training with all the data.

    Args:
        n_targets: The number of targets the GP should predict.
        fixed_noise: If True, use fixed noise GP, other wise use infered homoskedastic noise.

    Note: This is possible because `SingleTaskGP` does not fit the hyper parmas, that is done using
    >>> mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
    >>> fit_gpytorch_model(mll)

    Cons:
    - Dimesnion changes in shape
    - Less clear documentation

    Gotchas:
    - everything needs to be in model-space
    """
    train_x = get_train_x()
    n = train_x.shape[0]
    train_y = torch.sin(train_x).expand(n, n_targets)
    train_yvar = torch.ones_like(train_y) * 0.1

    model_original = SingleTaskGP(train_x, train_y, train_yvar if fixed_noise else None)

    # run the model once to avoid runtime error detailing this requirement
    _ = model_original.posterior(torch.tensor([[0.5]]))

    # Generate new training data to use
    new_x = torch.tensor([[0.5]])  # (n,d)
    new_y = torch.tensor([[0.1]]).expand(1, n_targets)  # (n, m)
    new_yvar = torch.ones_like(new_y) * 0.1

    # reshape the new_y and new_yvar to match the format of the underling model. This is required for this method
    # gpytorch docs say: (f,*b, m):
    # In botorch is is written as (*b,n') - which is not super clear because there is no m (target)
    # In reality this just need to match the dimensions of model.train_targets.
    # (n_targets,n) when n_targets> 1
    # (n,) when n_targets==1
    reshaped_new_y = new_y.reshape(-1) if n_targets == 1 else new_y.T
    reshaped_new_yvar = torch.ones_like(reshaped_new_y) * 0.1

    if fixed_noise:
        model_conditioned = model_original.get_fantasy_model(
            # gpytorch docs say: (f,*b, m, d). In botorch is is written as (*b,n',d)
            inputs=new_x,
            targets=reshaped_new_y,
            # This is Kwarg that gets passed through.  must match new_y
            noise=reshaped_new_yvar,
        )
    else:
        model_conditioned = model_original.get_fantasy_model(
            # gpytorch docs say: (f,*b, m, d). In botorch is is written as (*b,n',d)
            inputs=new_x,
            targets=reshaped_new_y,
        )

    # Expected model
    x_total = torch.vstack([train_x, new_x])  # (n+1,d=1)
    y_total = torch.vstack([train_y, new_y])  # (n+1,n_targets)
    yvar_total = torch.vstack([train_yvar, new_yvar])  # (n+1,n_targets)
    model_expected = SingleTaskGP(x_total, y_total, yvar_total if fixed_noise else None)

    # Check the internal data is the same
    check_same_model_data(model_actual=model_conditioned, model_expected=model_expected)
    # check that they both make the same preditiontion
    x = torch.tensor([[0.4], [0.7]])
    expected_posterior = model_expected.posterior(x)
    conditioned_posterior = model_conditioned.posterior(x)
    check_posterior(conditioned_posterior, expected_posterior)


@pytest.mark.external
@pytest.mark.filterwarnings("ignore: Input data is not standardized")
@pytest.mark.xfail(
    reason="Supplying noise in to update homoskedatic has side effects",
)
def test_get_fantasy_model_silent_failure():
    """Demonstate the silent failure when providing noise to a GP that doesn't use it.

    The posteriors produces are not the same. This is an edge case of `test_get_fantasy_model`.
    """
    fixed_noise = False
    n_targets = 1

    train_x = get_train_x()
    n = train_x.shape[0]
    train_y = torch.sin(train_x).expand(n, n_targets)
    train_yvar = torch.ones_like(train_y) * 0.1

    model_original = SingleTaskGP(train_x, train_y, train_yvar if fixed_noise else None)

    # run the model once to avoid runtime error detailing this requirement
    _ = model_original.posterior(torch.tensor([[0.5]]))

    # Generate new training data to use
    new_x = torch.tensor([[0.5]])  # (n,d)
    new_y = torch.tensor([[0.1]]).expand(1, n_targets)  # (n, m)
    new_yvar = torch.ones_like(new_y) * 0.1

    # reshape the new_y and new_yvar to match the format of the underling model. This is required for this method
    # gpytorch docs say: (f,*b, m):
    # In botorch is is written as (*b,n') - which is not super clear because there is no m (target)
    # In reality this just need to match the dimensions of model.train_targets.
    # (n_targets,n) when n_targets> 1
    # (n,) when n_targets==1
    reshaped_new_y = new_y.reshape(-1) if n_targets == 1 else new_y.T
    reshaped_new_yvar = torch.ones_like(reshaped_new_y) * 0.1

    # Shouldn't have noise, but provide it anyway
    model_conditioned = model_original.get_fantasy_model(
        # gpytorch docs say: (f,*b, m, d). In botorch is is written as (*b,n',d)
        inputs=new_x,
        targets=reshaped_new_y,
        # This is Kwarg that gets passed through.  must match new_y
        noise=reshaped_new_yvar,
    )

    # Expected model
    x_total = torch.vstack([train_x, new_x])  # (n+1,d=1)
    y_total = torch.vstack([train_y, new_y])  # (n+1,n_targets)
    yvar_total = torch.vstack([train_yvar, new_yvar])  # (n+1,n_targets)
    model_expected = SingleTaskGP(x_total, y_total, yvar_total if fixed_noise else None)

    # Check the internal data is the same
    check_same_model_data(model_actual=model_conditioned, model_expected=model_expected)
    # check that they both make the same preditiontion
    x = torch.tensor([[0.4], [0.7]])
    expected_posterior = model_expected.posterior(x)
    conditioned_posterior = model_conditioned.posterior(x)
    check_posterior(conditioned_posterior, expected_posterior)
