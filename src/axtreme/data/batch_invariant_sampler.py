"""Contains samplers where the total dataset procuded is not effect by the size of the batch dimension used.

Batch invariance in 1d.
This can be achieved using the standard BatchSampler, for example:

    >>> from torch.utils.data import BatchSampler
    >>> data = [1, 2, 3, 4, 5]
    >>> list(BatchSampler(data, batch_size=3, drop_last=False))
    [1, 2, 3], [4, 5, 6]]
    >>> list(BatchSampler(data, batch_size=2, drop_last=False))
    [[1, 2], [3, 4], [5]]

Regardless of batch size, these results can be concatenated along the batch dimension to prodcude the same result.

Batch invariance in 2d:

    >>> data = [1, 2, 3, 4, 5, 6]
    We want to turn things into a 2d dataset, where the dimension being batched along (e.g rows) does not effect the
    final data produced. This will not be the case with MultiBatchSampler:

    >>> list(MultiBatchSampler(data, batch_shape=torch.Size([2, 3])))
    [[
        [1, 2, 3],
        [4, 5, 6]
    ]]
    >>> b1, b2, b3 = list(MultiBatchSampler(data, batch_shape=torch.Size([2, 1])))
    >>> print(f"{b1=},{b2=},{b3=})
    b1= [[1], b2= [[3], b3= [[5],
         [2]]      [4]]      [6]]
    >>>  concat([b1,b2,b3], axis = -1)
    [[ 1, 3, 5]
     [ 2, 4, 6]]

The datasets produced will put data in different location based on the batch shape.
This is aproblem if you will be aggrgating (e.g over rows), and were expecting the batches to be invariant over that
dimension.

Provide invariante batching in a specific dimension.
"""
# pyright: reportUnnecessaryTypeIgnoreComment=false

from collections.abc import Iterable, Iterator

import torch
from torch.utils.data import Sampler


class BatchInvariantSampler2d(Sampler[list[list[int]]]):
    """Returns 2d batchs where the final dataset in invariant to changes in the last batch dimension.

    The standard BatchSampler
        - has 1 row that it batches in sizes b.
        - returns items of shape (1) * b
    This BatchSampler has:
        - n rows that are batched in size b.
        - returns batches of n * b

    Importantly, the concatenated batches (along the last dimension) will alway the same 2d matrix,
    regardless of the bach size.

    Examples:
        >>> Conceptually the full dataset might have the following indexs
        [[ 1, 3, 5]
         [ 2, 4, 6]]

        >>> If batched along index = -1, with batch size 2 will return results as follows
        b1= [[1,3],   b2= [[5],
             [2,4]]]       [6]]

        >>> concat([b1, b2], axis=-1)
        [[ 1, 3, 5]
         [ 2, 4, 6]]

    This is important when the final dataset produced should be invariate to changes in b.

    Note:
        - The final batch can return a partial batch (n rows, less than b columns)
        - Related to issue #76

    Todo:
        This is a very specific case because its cater to general case.
            - Can we make a more general 2d case? The batched dim needs to be filled last.
              Any aggregation is then expected on the batch dim
            - Can we make a general higher dim case?
            - Should we just take multiple samplers and batch them in parallel?
                - Pro: Might be cleaner conceptually
                - Con: When we treat as a dataset we want to get through all data before we start repeating data
    """

    def __init__(self, sampler: Sampler[int] | Iterable[int], batch_shape: torch.Size) -> None:
        """Will produce batch (along the rows) of a 2d batch.

        Args:
            sampler (Sampler[int] | Iterable[int]): Sampler (e.g RandomSampler or SerquentialSampler) to be batched.
                - Must be Sized (e.g have __len__)
                - See torch DataLoader implmentation for an examples
            batch_shape (torch.Size): The batch shape created of the underlieing sample.
        """
        self.sampler = sampler

        if len(batch_shape) != 2:  # noqa: PLR2004
            msg = f"Only 2d batches are supported, but recieved {batch_shape=}"
            raise ValueError(msg)
        self.batch_shape = batch_shape

        # Upfront check the the batch size is correct
        # don't want to fail on the last batch after spending a lot of time training
        # NOTE: see note about calling len(sampler) in method __len__
        data_len: int = len(sampler)  # type: ignore[arg-type]
        partial_batch_data_len = data_len % batch_shape.numel()
        if partial_batch_data_len % batch_shape[0] != 0:
            msg = (
                f"Final batch will have {partial_batch_data_len} items,"
                f" which does not fit into batch shape {torch.Size([*batch_shape[:-1] , -1])}"
            )
            raise ValueError(msg)

    def __len__(self) -> int:  # noqa: D105
        # NOTE: len is not a gaurenteed method on Sampler. but all the subclasses we use have it
        #       see note about calling len(sampler) in method __len__
        data_len: int = len(self.sampler)  # type: ignore[arg-type]
        return (data_len + self.batch_shape.numel() - 1) // self.batch_shape.numel()

    def __iter__(self) -> Iterator[list[list[int]]]:
        """Construct the indexes that will be in the batch.

        Returns:
            Returns a nested lists of the shape specified by `batch_shape`.
        """
        # Do all batches except the last, which might be partial batch
        sample_iter = iter(self.sampler)
        for _ in range(len(self) - 1):
            # the follow with throw StopIteration if there is not enough data. This should flow to the user.
            _batch: list[int] = [next(sample_iter) for _ in range(self.batch_shape.numel())]
            # By default fills across the row, we need it to fill down the columns
            # Fill the transpose of the batch shape, then transpose it back
            batch: torch.Tensor = torch.tensor(_batch).view(self.batch_shape[::-1]).T
            yield batch.tolist()

        # determine the shape of the final batch
        batch = torch.tensor(list(sample_iter)).reshape(-1, self.batch_shape[0]).T
        yield batch.tolist()
