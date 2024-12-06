"""Allows retuning batches of data of an arbitrary dimension."""
# pyright: reportUnnecessaryTypeIgnoreComment=false

from collections.abc import Iterable, Iterator

import torch
from torch.utils.data import Sampler


class MultiBatchSampler(Sampler[list[int]]):
    """Reads the entire sampler into batches of shape `batch_shape`.

    The final batch may not have enough samples to completely fill the batch shape. Behaviour is then as follows:

    - If `partial_batch_dim` is not None, attempt to batch samples allowing this dim to be variable in size. E.g:
        >>> batch_shape = torch.Size([3, 5])
        >>> partial_batch_index = -1
        try: remaining_samples.view([3,-1])

    This check is performed up front and the sampler will throw and error.

    Warning:
        - If the batch shape changes, even in the `partial_batch_index` dimension, data will be returned in a different
          order.
        - See BatchSampler2d for details.

    Todo:
        - Determing the right approach for handling partial batch that requires more than one parital index.
          e.g fit 5 items into batch_shape = torch.Size([3,2])
        - Perhaps return a new batch that is no bigger than the original in any dimension. This batcher is likely
          for gp throughput, so then the batches only have performance not logical meaning.
    """

    def __init__(
        self,
        sampler: Sampler[int] | Iterable[int],
        batch_shape: torch.Size,
        partial_batch_dim: None | int = -1,
    ) -> None:
        """Allows you to produces batchs of arbitrary shape.

        Args:
            sampler (Sampler[int] | Iterable[int]): Sampler (e.g RandomSampler or SerquentialSampler).

                - See torch DataLoader implementation for an examples

            batch_shape (torch.Size): The batch shape created of the underling sample.
            partial_batch_dim: Dimension of the batch that can be partially filled if there is not enough samples in
                sampler.

                - Currently only one dimension can be partially filled
                - if None, no dimension is allowed to be partially filled
        """
        self.sampler = sampler
        self.batch_shape = batch_shape
        self.partial_batch_dim = partial_batch_dim

        # Upfront check the the batch size is correct - don't want to fail after spending lots of time training
        # NOTE: see note about calling len(sampler) in method __len__
        data_len: int = len(sampler)  # type: ignore[arg-type]
        partial_batch_data_len = data_len % batch_shape.numel()
        if partial_batch_data_len != 0:
            if partial_batch_dim is None:
                msg = (
                    f"Final batch will have {partial_batch_data_len} items,"
                    f" which does not fit into batch shape {batch_shape}"
                )
                raise ValueError(msg)
            # Check the data perfectly fits into the specified partial batch shape.
            shape = list(batch_shape)
            shape[partial_batch_dim] = -1
            final_batch_shape = torch.Size(shape)
            if partial_batch_data_len % final_batch_shape.numel() != 0:
                msg = (
                    f"Final batch will have {partial_batch_data_len} items,"
                    f" which does not fit into batch shape {final_batch_shape}"
                )
                raise ValueError(msg)

    def __len__(self) -> int:  # noqa: D105
        # NOTE: len is not a guaranteed method on Sampler. but all the subclasses we use have it.
        #       see note about calling len(sampler) in method __len__
        data_len: int = len(self.sampler)  # type: ignore[arg-type]
        return (data_len + self.batch_shape.numel() - 1) // self.batch_shape.numel()

    def __iter__(self) -> Iterator[list[int]]:
        """Returns a nested lists of the shape specified by `batch_shape`."""
        # Do all batches except the last, which might be partial batch
        sample_iter = iter(self.sampler)
        for _ in range(len(self) - 1):
            # the follow with throw StopIteration if there is not enough data. This should flow to the user.
            batch = [next(sample_iter) for _ in range(self.batch_shape.numel())]
            yield torch.tensor(batch).view(self.batch_shape).tolist()

        # determine the shape of the final batch
        final_batch_shape = self.batch_shape
        if self.partial_batch_dim is not None:
            shape = list(self.batch_shape)
            shape[self.partial_batch_dim] = -1
            final_batch_shape = torch.Size(shape)

        yield torch.tensor(list(sample_iter)).reshape(final_batch_shape).tolist()
