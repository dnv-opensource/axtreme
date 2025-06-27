"""Test for gp_bruteforce."""

# %%
from collections.abc import Callable

import matplotlib.pyplot as plt
import pytest
import torch
from botorch.models import SingleTaskGP
from botorch.models.deterministic import GenericDeterministicModel, PosteriorMeanModel
from botorch.models.ensemble import EnsembleModel
from botorch.sampling import IIDNormalSampler
from botorch.sampling.index_sampler import IndexSampler
from tests.qoi.utils import dummy_posterior_mean

from axtreme.plotting.gp_fit import plot_1d_model
from axtreme.qoi.gp_bruteforce import GPBruteForce
from axtreme.sampling import NormalIndependentSampler, PosteriorSampler, UTSampler


class TestSampleSurrogate:
    """Sammple_surrogate: What do we want to test:
    - works with just a single dimension
        - 1 sample
    - works with multiple dimensions (liek the one we will probably use in the problem)
        - One or 2 samples
    - can batch:
        - actual example we intend to use
    """

    def test_single_param(self):
        """Check that the sample_surrogate works with non-batched params and a single sample.

        Because scale is 0, the sample should be equal to the loc.
        """
        loc = torch.tensor([1.0])
        scale = torch.tensor([0.0])

        params = torch.cat([loc, scale], dim=-1)

        sample = GPBruteForce.sample_surrogate(params)

        assert sample == pytest.approx(1.0, abs=1e-5)

    def test_batched_params(self):
        """Check that the sample_surrogate works with batched params and a single sample.

        Because scale is 0, the sample should be equal to the loc. Dimensionality is chosen to mimic what this function
        is commonly used for.
        """
        # Shape represents (n_posterior_samples = 2, n_periods = 3, batch_size = 5, 1)
        # fmt:off
        loc = torch.tensor([
            [[[ 0.],  [1.], [ 2.], [ 3.],  [4.]],
            [[ 5.],  [6.], [ 7.], [ 8.],  [9.]],
            [[10.], [11.], [12.], [13.], [14.]]],

            [[[15.], [16.], [17.], [18.], [19.]],
            [[20.], [21.], [22.], [23.], [24.]],
            [[25.], [26.], [27.], [28.], [29.]]]])
        # fmt: on
        scale = torch.zeros_like(loc) * 0.0

        # Shape represents (n_posterior_samples = 2, n_periods = 3, batch_size = 5, p = 2)
        params = torch.cat([loc, scale], dim=-1)

        sample = GPBruteForce.sample_surrogate(params)

        # shape should be ( number of samples, *b)
        assert sample.shape == torch.Size([1, 2, 3, 5])

        sample1 = sample[0]
        # because scale is 0, the sample should be equal to the loc (need to make the dimension right)
        expected_value = loc.squeeze(-1)
        torch.testing.assert_close(sample1, expected_value, atol=1e-5, rtol=0)

    def test_batched_params_multiple_samples(self):
        """Exactly the same as test__batched_params, but 2 samples are drawn.

        Because scale is 0, the sample should be equal to the loc, and both samples should be the same
        """
        # Shape represents (n_posterior_samples = 2, n_periods = 3, batch_size = 5, 1)
        # fmt:off
        loc = torch.tensor([
            [[[ 0.],  [1.], [ 2.], [ 3.],  [4.]],
            [[ 5.],  [6.], [ 7.], [ 8.],  [9.]],
            [[10.], [11.], [12.], [13.], [14.]]],

            [[[15.], [16.], [17.], [18.], [19.]],
            [[20.], [21.], [22.], [23.], [24.]],
            [[25.], [26.], [27.], [28.], [29.]]]])
        # fmt: on
        scale = torch.zeros_like(loc) * 0.0

        # Shape represents (n_posterior_samples = 2, n_periods = 3, batch_size = 5, p = 2)
        params = torch.cat([loc, scale], dim=-1)

        sample = GPBruteForce.sample_surrogate(params, n_samples=2)

        # shape should be ( number of samples, *b)
        assert sample.shape == torch.Size([2, 2, 3, 5])

        # because scale is 0, the sample should be equal to the loc (need to make the dimension right)
        expected_value = loc.squeeze(-1)
        torch.testing.assert_close(sample[0], expected_value, atol=1e-5, rtol=0)
        torch.testing.assert_close(sample[1], expected_value, atol=1e-5, rtol=0)

    # NOTE: strictly speaking this is non-deterministic, but it should never cause issues.
    def test_shared_base_samples(self):
        """When base samples are shared, the same quantil should be used across the batching demension.

        The following example is close to the usecase for this. Both posteriors should get the same base samples.

        We check this by ensuring corresponsing sample in the two postiors show the same quantile. We use the same scale
        across both posteriors, so we test this by checking they have the same offset from their mean (given by loc).

        Because scale is 0, the sample should be equal to the loc, and both samples should be the same
        """
        # Shape represents (n_posterior_samples = 2, n_periods = 3, batch_size = 5, 1)
        # fmt:off
        loc = torch.tensor([
            [[[ 0.],  [1.], [ 2.], [ 3.],  [4.]],
            [[ 5.],  [6.], [ 7.], [ 8.],  [9.]],
            [[10.], [11.], [12.], [13.], [14.]]],

            [[[15.], [16.], [17.], [18.], [19.]],
            [[20.], [21.], [22.], [23.], [24.]],
            [[25.], [26.], [27.], [28.], [29.]]]])
        # fmt: on
        scale = torch.ones_like(loc) * 1.0

        # Shape represents (n_posterior_samples = 2, n_periods = 3, batch_size = 5, p = 2)
        params = torch.cat([loc, scale], dim=-1)

        sample = GPBruteForce.sample_surrogate(params, n_samples=1, base_sample_broadcast_dims=[0])

        # shape should be ( number of samples, *b)
        assert sample.shape == torch.Size([1, 2, 3, 5])

        # remove the excess dimensions from each
        sample = sample.squeeze(0)
        loc = loc.squeeze(-1)
        # remove the sampling dimension and subtract the element wise mean (using loc)
        offset_posterior1 = sample[0] - loc[0]
        offset_posterior2 = sample[1] - loc[1]

        torch.testing.assert_close(offset_posterior1, offset_posterior2, atol=1e-5, rtol=0)

        # NOTE: Samples should only be share on the n_posterior sample dim. Across other dims things should be unique.
        assert len(offset_posterior1.unique()) == offset_posterior1.numel()


class TestProcessBatch:
    """There is no a large amount of out own code to cover here.

    Integration tests:
    - Check it works with sample_surrogate

    Todo:
        - test with posterior transforms
        - test with posterior the have sample shape that is multidim.
    """

    @pytest.mark.integration
    def test__process_batch(
        self, gp_passthrough_1p: GenericDeterministicModel, gp_passthrough_1p_sampler: IndexSampler
    ):
        """There is a low amount of our own code in"""
        qoi = GPBruteForce(
            env_iterable=[],  # Shouldn't be used in this test
            posterior_sampler=gp_passthrough_1p_sampler,
        )

        # fmt:off
        # Shape is: (n_periods = 3, n_samples_per_period = 5, d = 1)
        env_batch = torch.tensor(
            [
                    [[0.0], [1.0], [2.0], [3.0], [4.0]],
                    [[5.0], [6.0], [7.0], [8.0], [9.0]],
                    [[10.0], [11.0], [12.0], [13.0], [14.0]],

            ]
        )
        # fmt: on

        # Run test
        # expected shape (erd_samples_per_period = 1, n_posterior_samples = 1, n_periods = 3)
        actual_result = qoi._process_batch(env_batch, gp_passthrough_1p)

        # The largest from each row should be selected (in the shape above)
        expected_result = torch.tensor([[[4.0, 9.0, 14.0]]])

        # some relative tollerance allowed as we sample from a very confident gumbel dist
        torch.testing.assert_close(actual_result, expected_result, atol=1e-4, rtol=0)


def env_period_1() -> torch.Tensor:
    """Helper to creates a single period env sample.

    typical shape: (periods, n_samples_per_period, d)

    NOTE: changes here will break test_posterior_samples_erd_samples

    Returns:
    Shape: (1, 5, 1)
    """
    # fmt: off
    env_samples = torch.tensor(
        [
            [[0], [1], [2], [3], [4]],
        ]
    )
    # fmt: on
    return env_samples


def env_period_2() -> torch.Tensor:
    """Helper to creates an env sample with 2 periods.

    typical shape: (periods, n_samples_per_period, d)

    NOTE: changes here will break test_posterior_samples_erd_samples

    Returns:
    Shape: (2, 45 1)
    """
    # fmt: off
    env_samples = torch.tensor(
        [
            [[0], [1], [2], [3], [4]],
            [[9], [8], [7], [6], [5]]
        ]
    )
    # fmt: on
    return env_samples


def batched_env_period() -> list[torch.Tensor]:
    """Batched env data.

    Batching turns;
    env_samples = torch.tensor(
        [
            [[0], [1], [2], [3], [4]], # period1
            [[9], [8], [7], [6], [5]]  # period2
        ]
    )
    Into:
    [
        [
            [[0], [1]],
            [[9], [8]]
        ],
        [
            [[2], [3]],
            [[7], [6]]
        ],
        [
            [[4]],
            [[5]]
        ]
    ]
    """
    # fmt: off
    env_samples = torch.tensor(
        [
            [[0], [1], [2], [3], [4]], # period1
            [[9], [8], [7], [6], [5]]  # period2
        ]
    )
    # fmt: on
    return list(torch.chunk(env_samples, 3, dim=-2))


# TODO(sw 2024-11-22): Now that the function has been refactored this test could be broken up by mocking _process_batch
@pytest.mark.parametrize(
    "env_iterable, n_posterior_samples, erd_samples_per_period, expected_output",
    [
        pytest.param([env_period_1()], 1, 1, torch.tensor([[4]]), id="single_everything"),
        pytest.param([env_period_2()], 1, 1, torch.tensor([[4, 9]]), id="n_period_2"),
        pytest.param([env_period_1()], 2, 1, torch.tensor([[4], [4]]), id="n_posterior_samples_2"),
        pytest.param([env_period_1()], 1, 2, torch.tensor([[4, 4]]), id="erd_samples_2"),
        # Should produce the exact same results as n_period_2 test above
        pytest.param(batched_env_period(), 1, 1, torch.tensor([[4, 9]]), id="batched_env"),
    ],
)
def test_posterior_samples_erd_samples(
    env_iterable: list[torch.Tensor],
    n_posterior_samples: int,
    erd_samples_per_period: int,
    expected_output: torch.Tensor,
    gp_passthrough_1p: GenericDeterministicModel,
):
    """Checks the ERD samples for different configurations of GPBruteForce.

    By fixed the model, posterior samples, and gumbel samples produced, we can determinsitically trace input values
    through to their expect value in the erd samples. This is done by using `dummy_posterior_mean` .

    Specifically this tests:
    - Single sample of all params returns the 1 ERD sample.
    - multiple periods: each one generates and ERD sample.
    - multiple posterior samples: each generate a set of ERD samples
    - multiple erd_samples_per_period: multiple ERD samples are generated per period
    - Batching: doesn't effect the ERD generated

    Args:
        env_iterable: can be of the following shape;
            - non-batched shape: (1, periods, n_samples_per_period, d), where d=1
            -  batched shape: (n_batches, periods, n_samples_per_period//n_batches, d), d=1
        n_posterior_samples: number of posterior samples taken
        erd_samples_per_period: number of erd samples taken
        expected_output: output of shape (n_posterior_samples, total_erd_samples) expected.
    """
    # Create GP like object around dummy_posterior_mean
    model = gp_passthrough_1p
    # Sampler that belongs with DeterministicModels
    sampler = IndexSampler(torch.Size([n_posterior_samples]))

    qoi = GPBruteForce(env_iterable, posterior_sampler=sampler, erd_samples_per_period=erd_samples_per_period)

    # produces output of shape (n_posterior_samples, n_periods * erd_samples_per_period)
    ests = qoi.posterior_samples_erd_samples(model)

    # using atol = 1e-5 covers the entire support of Gumbel(loc = x, scale = 1e-6)
    torch.testing.assert_close(ests, expected_output, atol=1e-5, rtol=0, check_dtype=False)


# %%
def precalculate_2_posterior_samples(X: torch.Tensor) -> torch.Tensor:  # noqa: N803
    """Creates 2 posterior samples, where sample2 = sample1 + 10.

    Using this in conjuction with:
        - GenericDeterministicModel: to create an Ensemble model
        - IndexSampler with base_samples = torch.tensor([0,1])

    Args:
    X: typically (n_periods, n, d)

    Return:
    (*b,s,n,t):
        - *b batch dimension
        - s: number of ensembels (e.g number of samples at point x)
        - n: number of input points
        - t: number of targets
    """
    posterior1 = dummy_posterior_mean(X)
    posterior2 = dummy_posterior_mean(X)
    # increase the loc value
    posterior2[..., 0] += 10

    # this is shape (s, n_periods, n, t)
    output = torch.stack([posterior1, posterior2])
    # Swap it to be of the required form *b,s,n,t)
    return torch.swapaxes(output, 0, 1)


class GenericEnsembleModel(EnsembleModel):
    r"""A generic ensembel model constructed from a callable."""

    def __init__(self, f: Callable[[torch.Tensor], torch.Tensor]) -> None:
        """Create an Ensemble model from an underling function.

        Args:
            f: A callable mapping a `batch_shape x n x d`-dim input tensor `x`
                to a `batch_shape x s x n x m`-dimensional output tensor (the
                outcome dimension `m` must be explicit, even if `m=1`). `s` is the number
        """
        super().__init__()
        self._f = f

    def forward(self, X: torch.Tensor) -> torch.Tensor:  # noqa: N803 # for consitence with botorch interface
        r"""Compute the (deterministic) model output at X.

        Args:
            X: A `batch_shape x n x d`-dim input tensor `X`.

        Returns:
            A `batch_shape x n x m`-dimensional output tensor.
        """
        return self._f(X)


def test_posterior_samples_erd_samples_grouped_by_posterior():
    """Ensure that ERD samples are listed under the posterior that generated them.

    The mocked gp and posterior combination means that posterior samples produced for loc will be:
    [
        # posterior sample 2
        [
            [[0], [1], [2], [3], [4]],
            [[9], [8], [7], [6], [5]]
        ],
        # posterior sample 2
        [
            [[10], [11], [12], [13], [14]],
            [[19], [18], [17], [16], [15]]
        ]
    ]
    Beause the scale is ~0, the Gumbel samples will be the same as the loc values.
    """
    # fmt: off
    env_data = torch.tensor(
        [
            [[0], [1], [2], [3], [4]],
            [[9], [8], [7], [6], [5]]
        ]
    )
    # fmt: on

    # NOTE: The combination of precalculate_2_posterior_samples, GenericEnsembleModel, IndexSampler has been tested
    # manually to check it produces the required output
    model = GenericEnsembleModel(precalculate_2_posterior_samples)

    # Chose the order that these are sampled in
    sampler = IndexSampler(torch.Size([2]))
    sampler.base_samples = torch.tensor([0, 1])

    qoi = GPBruteForce([env_data], posterior_sampler=sampler)

    # produces output of shape (n_posterior_samples, n_periods * erd_samples_per_period)
    ests = qoi.posterior_samples_erd_samples(model)

    # fmt: off
    expected_output = torch.tensor(
        [
            [4,9],
            [14,19]
        ]
    )
    # fmt: on

    # using atol = 1e-5 covers the entire support of Gumbel(loc = x, scale = 1e-6)
    torch.testing.assert_close(ests, expected_output, atol=1e-5, rtol=0, check_dtype=False)


def test_grad():
    # env_iterable and posterior sampler are not used in this test
    with pytest.raises(NotImplementedError):
        _ = GPBruteForce(env_iterable=[], posterior_sampler=IndexSampler(torch.Size([1])), no_grad=False)


def test_gpu():
    # env_iterable and posterior sampler are not used in this test
    with pytest.raises(NotImplementedError):
        _ = GPBruteForce(env_iterable=[], posterior_sampler=IndexSampler(torch.Size([1])), device=torch.device("cuda"))


# NOTE: This functionaltiy is important for GPBruteForce, so the test is put here, but could in in a number of places.
@pytest.mark.integration
@pytest.mark.parametrize(
    "sampler",
    [
        pytest.param(NormalIndependentSampler(torch.Size([5]), seed=7), id="NormalIndependentSampler"),
        pytest.param(UTSampler(), id="UTSampler"),
        pytest.param(
            IIDNormalSampler(torch.Size([5]), seed=7),
            marks=pytest.mark.xfail(reason="Demonstration of non-batchable posterior"),
            id="IIDNormalSampler (from botorch)",
        ),
    ],
)
def test_posterior_samples_are_equivalent_when_batched(
    sampler: PosteriorSampler,
    model_singletaskgp_d1_t1: SingleTaskGP,
    *,
    visual_inspect: bool = False,  # noqa: PT028
):
    """Tests if posterior samplers are invariant to batching on input data as is required for GPBruteForce.

    Often we work with periods of env data that are too large to predict a single posterior for. In this case it is
    useful to batch the data, and run the GP on each batch.One shortcoming is that the covariance between items in
    different batches is lost. Posterior samplers that make use of the covariance between different points will produce
    different samples depending on if the data is batched or not. This can be problematic for the following reasons:
    - Ideally the result should be invariant to the batch dimension choosen.
    - In the batch case the posterior samples will be discontinous, which can mean samples no longer reflect behviour of
      the underling function being modeled.

    The following compares the posterior generate on non-batched and batched data, and checks if they are the same.
    `visual_inspect` can be used when running this function manually to visually check the results.
    """
    model = model_singletaskgp_d1_t1

    #### None of following
    env_data = torch.linspace(0, 1, 100).reshape(-1, 1)

    # not batched process
    env_iterable = [env_data]

    for data in env_iterable:
        with torch.no_grad():
            posterior = model.posterior(data)

        samples = sampler(posterior)
        samples = samples.squeeze(-1)  # flatten the t=1 dimension

    if visual_inspect:
        ax = plot_1d_model(model)
        for sample in samples:
            _ = ax.plot(env_data.flatten(), sample, color="grey")
        plt.show()

    # Batch processing
    batch_env_iterable = torch.chunk(env_data, 3)

    batches = []
    for data in batch_env_iterable:
        with torch.no_grad():
            posterior = model.posterior(data)
        batch_samples = sampler(posterior)
        batch_samples = batch_samples.squeeze()
        batches.append(batch_samples)

    if visual_inspect:
        ax = plot_1d_model(model)
        colour = ["r", "g", "b"]
        for batch_x, batch_samples, c in zip(batch_env_iterable, batches, colour, strict=True):
            for sample in batch_samples:
                _ = ax.plot(batch_x.flatten(), sample, color=c)
        plt.show()

    # Check the batch sample and the full samples are the same once batches are combined
    # batches have dimensions (n_posterior_samples, batch_size).
    # Combine the batches so get back to shape (n_posterior_samples, env_samples)
    joined_batches = torch.cat(batches, dim=-1)
    torch.testing.assert_close(samples, joined_batches)


def test_shared_surrogate_base_samples_no_gp_uncertainty(model_singletaskgp_d1_t2: SingleTaskGP):
    """With shared surrogate bases sample and not GP uncertainty, all qoi outputs should be identical.

    When there is no GP uncertainty, all posterior samples are identical. This means they output the exact same
    distributions at each x point. When base samples are shared across posteriors, the exact sample point/quantile is
    sampled from each of the identical distributions. As a result all QoI posteriors should produce the same result.
    """

    model = PosteriorMeanModel(model_singletaskgp_d1_t2)
    sampler = IndexSampler(torch.Size([10]))

    # (n_periods, period_len, d)
    env_data = torch.rand(3, 5, 1)

    qoi = GPBruteForce([env_data], posterior_sampler=sampler, shared_surrogate_base_samples=True)

    results = qoi(model)

    assert torch.all(results == results[0])
