import matplotlib.pyplot as plt
import pytest
import torch
from botorch.models import SingleTaskGP

from axtreme.plotting.gp_fit import plot_1d_model
from axtreme.sampling import NormalIndependentSampler, PosteriorSampler, UTSampler


@pytest.mark.integration
@pytest.mark.parametrize(
    "sampler, expected_value",
    [
        (
            NormalIndependentSampler(sample_shape=torch.Size([5]), seed=7),
            torch.tensor(
                [
                    [[0.5811], [0.0562], [-0.9330], [0.0768], [-0.4511]],
                    [[1.2794], [0.8726], [-0.6471], [0.8613], [-0.1794]],
                    [[1.3997], [1.0132], [-0.5979], [0.9965], [-0.1326]],
                    [[-0.1431], [-0.7904], [-1.2295], [-0.7369], [-0.7330]],
                    [[1.9565], [1.6641], [-0.3699], [1.6221], [0.0841]],
                ]
            ),
        ),
        (
            UTSampler(),
            torch.tensor(
                [
                    [[0.6910], [0.1847], [-0.8881], [0.2002], [-0.4084]],
                    [[1.9874], [1.7002], [-0.3573], [1.6568], [0.0962]],
                    [[-0.6054], [-1.3309], [-1.4188], [-1.2564], [-0.9129]],
                ]
            ),
        ),
    ],
)
def test_sampler_integration(
    sampler: PosteriorSampler,
    expected_value: torch.Tensor,
    model_singletaskgp_d1_t1: SingleTaskGP,
    *,
    visual_inspect: bool = False,
):
    """One of the most useful ways to inspect the performance of a sampler is to visually inspect it.

    This test confirms the output matches the visual inspecation (expected_value come from the viual inspection.

    Args:
        sampler: _description_
        expected_value: _description_
        model_singletaskgp_d1_t1: _description_
        visual_inspect: _description_. Defaults to False.

    Todo:
        - A posterior could be passed directly here instead of the model. model has been use so different vidualisations
          can easily be made.
    """
    model = model_singletaskgp_d1_t1
    x = torch.linspace(0, 1, 5).reshape(-1, 1)

    with torch.no_grad():
        posterior = model.posterior(x)

    samples = sampler(posterior)

    if visual_inspect:
        # For better visual inspection it is more useful to increase the number of points in x
        ax = plot_1d_model(model)
        for sample in samples:
            _ = ax.plot(x.flatten(), sample, color="grey")
        plt.pause(5)

    torch.testing.assert_close(samples, expected_value, rtol=1e-3, atol=1e-4)
