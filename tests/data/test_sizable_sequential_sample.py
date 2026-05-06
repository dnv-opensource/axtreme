import numpy as np

from axtreme.data import MinimalDataset
from axtreme.data.sizable_sequential_sample import SizableSequentialSampler


def test_SizableSequentialSampler_less_sample_than_data():
    minimal_dataset = MinimalDataset(np.arange(8))

    seq_samper = SizableSequentialSampler(minimal_dataset, num_samples=4)
    assert list(seq_samper) == [0, 1, 2, 3]


def test_SizableSequentialSampler_from_data():
    minimal_dataset = MinimalDataset(np.arange(8))

    seq_samper = SizableSequentialSampler(minimal_dataset)
    assert list(seq_samper) == [0, 1, 2, 3, 4, 5, 6, 7]


def test_SizableSequentialSampler_more_sample_than_data():
    minimal_dataset = MinimalDataset(np.arange(8))

    seq_samper = SizableSequentialSampler(minimal_dataset, num_samples=10)
    assert list(seq_samper) == [0, 1, 2, 3, 4, 5, 6, 7, 0, 1]


def test_SizableSequentialSampler_stateful_advances_offset():
    """When stateful=True, each iteration continues from where the last ended."""
    minimal_dataset = MinimalDataset(np.arange(8))

    sampler = SizableSequentialSampler(minimal_dataset, num_samples=3, stateful=True)
    assert list(sampler) == [0, 1, 2]
    assert list(sampler) == [3, 4, 5]
    assert list(sampler) == [6, 7, 0]  # wraps around
    assert list(sampler) == [1, 2, 3]


def test_SizableSequentialSampler_stateful_false_resets():
    """When stateful=False (default), every iteration starts from index 0."""
    minimal_dataset = MinimalDataset(np.arange(8))

    sampler = SizableSequentialSampler(minimal_dataset, num_samples=3, stateful=False)
    assert list(sampler) == [0, 1, 2]
    assert list(sampler) == [0, 1, 2]


def test_SizableSequentialSampler_stateful_more_samples_than_data():
    """Stateful mode wraps correctly when num_samples > len(data_source)."""
    minimal_dataset = MinimalDataset(np.arange(4))

    sampler = SizableSequentialSampler(minimal_dataset, num_samples=6, stateful=True)
    assert list(sampler) == [0, 1, 2, 3, 0, 1]
    assert list(sampler) == [2, 3, 0, 1, 2, 3]
