import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, SequentialSampler

from axtreme.data import MinimalDataset, MultiBatchSampler

# what do we want to check on the multibatch sampler?

# Bad batch shape give an error
# That the content of the batchs changes with different sample size


# Usecase tests: (this well go in a demofile)
# Have a lot of data, use a subset to make something and iterate though
# if you intend to combine it show that should use the other one
# What do do if you want it iterate through a whole dataset, and want to do it fast.
# (recommended just make it fit the batch shape for now)


@pytest.mark.parametrize(
    "batch_shape",
    [
        torch.Size([12]),
        torch.Size([6]),
        torch.Size([4, 3]),
        torch.Size([2, 3]),
        torch.Size([2, 3, 2]),
    ],
)
def test_multibatchsampler_full_batchs(batch_shape: torch.Size):
    # sourcery skip: no-loop-in-tests
    # Typically this will be a Sampler object, but an iterator is perfectly valid, and is simpler to is used here
    samples = list(range(12))
    batch_sampler = MultiBatchSampler(samples, batch_shape=batch_shape)

    for batch in batch_sampler:
        assert torch.tensor(batch).shape == batch_shape


@pytest.mark.parametrize(
    "batch_shape, partial_batch_dim",
    [
        # 1d check
        (torch.Size([5]), -1),  # partial batch of size torch.Size([2])
        # 2d check
        (torch.Size([2, 5]), -1),  # partial batch of size torch.Size([2,1])
        (torch.Size([5, 2]), 0),  # partial batch of size torch.Size([1,2])
        # 3d check
        (torch.Size([2, 2, 2]), -1),  # partial batch of size torch.Size([2,2,1])
        (torch.Size([2, 2, 2]), 1),  # partial batch of size torch.Size([2,1,2])
    ],
)
def test_multibatchsampler_partial_batchs(batch_shape: torch.Size, partial_batch_dim: int):
    # sourcery skip: no-loop-in-tests
    # Typically this will be a Sampler object, but an iterator is perfectly valid, and is simpler to is used here
    samples = list(range(12))
    batch_sampler = MultiBatchSampler(samples, batch_shape=batch_shape, partial_batch_dim=partial_batch_dim)

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

    del batch_shape_list[partial_batch_dim]
    del partial_batch_shape_list[partial_batch_dim]

    assert batch_shape_list == partial_batch_shape_list


@pytest.mark.parametrize(
    "batch_shape, data_length,  partial_batch_dim",
    [
        # 1d check
        (torch.Size([5]), 12, None),
        # # 2d check
        (torch.Size([5, 2]), 12, -1),  # partial batch of size torch.Size([2,1])
        (torch.Size([2, 5]), 12, 0),  # partial batch of size torch.Size([1,2])
    ],
)
def test_multibatchsampler_incompatible_partial_batchs(
    batch_shape: torch.Size, data_length: int, partial_batch_dim: int
):
    """We know ahead of time when the partial batch will fail, because we only allow one dime to be parial.

    e.g if trying to fit [1,2,3,4,5] into shape
        [[.,.],
         [.,.],
         [.,.]]
    """
    samples = list(range(data_length))

    with pytest.raises(ValueError, match="..* does not fit into batch shape,*"):
        _ = MultiBatchSampler(samples, batch_shape=batch_shape, partial_batch_dim=partial_batch_dim)


def test_multibatchsampler_no_batch_invariant():
    """Samples in batchs are not invariant to the partial_dim (which is typically the on batching along).

    The follow show how the data along the partial_dimension (rows) changes depending on the size.
    This will cause variance between runs with different batch size if you later operate along this dim (e.g rows).

    See BatchSampler2dInvariant for details.
    """
    samples = list(range(8))
    # What ould happen if full size batching was used
    batch_sampler2 = MultiBatchSampler(samples, batch_shape=torch.Size([2, 4]))
    samples2 = torch.concat([torch.tensor(b) for b in batch_sampler2], dim=-1)

    expected_result2 = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])

    torch.testing.assert_close(samples2, expected_result2)

    # What will happen when batch dimension is changed

    batch_sampler1 = MultiBatchSampler(samples, batch_shape=torch.Size([2, 2]))
    samples1 = torch.concat([torch.tensor(b) for b in batch_sampler1], dim=-1)

    expected_result1 = torch.tensor([[0, 1, 4, 5], [2, 3, 6, 7]])

    torch.testing.assert_close(samples1, expected_result1)


### Integration test
def test_multibatchsampler_integration_compatible_with_multiworker():
    """Integration test to check that works with the standard components *e.g Sampler and Dataloader.

    NOTE: any call starts iterating through a dataloader object, need to be inside __name__=="__main__"
    (more specifically, must not run on import). This stops the subprocess for running it again (creating a whole new
    set of worker).
    IF you set num_workers = 0 then don't need to wory about this as work is just done in current process.
    """
    # sourcery skip: no-loop-in-tests, remove-empty-nested-block
    # make a fake dataseet where each feature has 2 dims
    data = np.arange(0, 200).reshape(-1, 2)
    ds = MinimalDataset(data)
    sampler = SequentialSampler(ds)

    batch_sampler = MultiBatchSampler(sampler, batch_shape=torch.Size([4, 3]))
    dl = DataLoader(ds, batch_sampler=batch_sampler, num_workers=2)

    for _ in dl:
        pass
