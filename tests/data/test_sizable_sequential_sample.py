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
