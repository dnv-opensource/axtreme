"""These items typically are part of creating fixture in conftest.py, but are also imported directly in the test.

Placed here as pytest compains when importing from conftest directly.
"""

import torch


def dummy_posterior_mean(x: torch.Tensor) -> torch.Tensor:
    """This is a helper function to be used with GenericDeterministicModel to create models with deterministic output.

    This can be created with the following:
    > model = GenericDeterministicModel(dummpy_posterior_mean, num_outputs=2)

    The behaviour of this function is to create:
        - `loc` is set equal to the `env_value`
        - `scale` is set to 1e-6

    Useful because the result of sampling from the subsequent gumbel distibution is effectively deterministic.

    Args:
    x: (*b,d=1): a batch of env data, where each env datapoint has dim=1

    Return:
    (b*, 2): where the last dimension is [loc, scale]
    """
    scale = torch.ones_like(x) * 1e-6
    return torch.concat([x, scale], dim=-1)
