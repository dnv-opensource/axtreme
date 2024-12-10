# What are the key things to check?

# only support 2d
# that the order doesnt change with different batch sizes

import pytest
import torch

from axtreme.data.batch_invariant_sampler import BatchInvariantSampler2d


@pytest.mark.parametrize(
    "batch_shape",
    [
        # 2d check
        (torch.Size([2, 5])),  # partial batch of size torch.Size([2,1])
    ],
)
def test_batch_invariant_sampler_partial_batchs(batch_shape: torch.Size):
    # Typically this will be a Sampler object, but an iterator is perfectly valid, and is simpler to is used here
    # sourcery skip: no-loop-in-tests
    samples = list(range(12))
    batch_sampler = BatchInvariantSampler2d(samples, batch_shape=batch_shape)

    # Check the standard batches
    batches = list(batch_sampler)
    for batch in batches[:-1]:
        # print this if want more viual debugging
        assert torch.tensor(batch).shape == batch_shape

    # check the the final batch has the same dimensionf for all but the partial_batch_dim
    partial_batch = batches[-1]
    partial_batch_shape = torch.tensor(partial_batch).shape
    # remove the partial index from each and check that shape matches otherwise
    batch_shape_list = list(batch_shape)
    partial_batch_shape_list = list(partial_batch_shape)

    del batch_shape_list[-1]
    del partial_batch_shape_list[-1]

    assert batch_shape_list == partial_batch_shape_list


def test_batch_invariant_sampler_no_batch_invariant():
    samples = list(range(8))
    # What ould happen if full size batching was used
    batch_sampler2 = BatchInvariantSampler2d(samples, batch_shape=torch.Size([2, 4]))
    samples2 = torch.concat([torch.tensor(b) for b in batch_sampler2], dim=-1)

    # What will happen when batch dimension is changed
    batch_sampler1 = BatchInvariantSampler2d(samples, batch_shape=torch.Size([2, 2]))
    samples1 = torch.concat([torch.tensor(b) for b in batch_sampler1], dim=-1)

    torch.testing.assert_close(samples1, samples2)


def test_batch_invariant_sampler_not_2d():
    samples = list(range(12))

    with pytest.raises(ValueError, match=".*Only 2d batches are supported.*"):
        _ = BatchInvariantSampler2d(samples, batch_shape=torch.Size([1, 2, 3]))
