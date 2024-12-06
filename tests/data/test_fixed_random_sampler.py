import numpy as np

from axtreme.data.dataset import MinimalDataset
from axtreme.data.fixed_random_sample import FixedRandomSampler


def test_indentical_random_sample_over_multiple_iterations():
    """Check that each time the sampler is iterated over it produces identical samples.

    This is what makes the dataloader produce identical data each time it it iterated over (in its entirety).
    """
    data = np.arange(5).reshape(-1, 1)
    dataset = MinimalDataset(data)

    sampler = FixedRandomSampler(dataset, num_samples=10, seed=0, replacement=False)

    iteration1 = list(sampler)
    iteration2 = list(sampler)

    assert iteration1 == iteration2
