"""Contains objects and helpers for creating data unit testing."""

from collections.abc import Sequence
from typing import Any, TypeVar, cast

import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset

# TODO(sw): Dataset makes the type covariant (e.g TypeVar('T_co', covariant=True)).
# Struggling to understand contravariate/covariant/invariant. Come back to this.
T = TypeVar("T", Sequence[Any], NDArray[Any], torch.Tensor)


class MinimalDataset(Dataset[T]):
    """Creates a Dataset over a list-like data source.

    Note:
        `list`, `np.array` or `tensor` forfill the `__getitem__` and `__len__` requirement directly - but this makes
        them conform with the __add__ behaviour defined for datasets.
    """

    def __init__(self, data: Sequence[T] | T) -> None:  # . D417
        """Creates a Dataset over a list-like data source.

        Args:
            data: Supports being indexed (supports __getitem__) and len, where index values are considered datapoints.
            - e.g (n,d) numpy dataset, a List[datapoint].
            - Datapoints should be compatible with DataLoader.
                - General work with numerics and matrixes
                - Specifics of what DataLoader consume from datasets see `torch._utils.collate.default_collate`

        Examples:
            >>> data = [1, 2, 3]
            >>> ds = MinimalDataset(data)
            >>> ds[0]
            1

            >>> data = np.array([[1, 2, 3], [4, 5, 6]])
            >>> ds = MinimalDataset(data)
            >>> ds[0]
            1p.array([1,2,3])

            >>> data = torch.tensor([[1], [2]])
            >>> ds = MinimalDataset(data)
            >>> ds[0]
            torch.tensor([1])
        """
        self.data: Sequence[T] | T = data

    def __len__(self) -> int:  # noqa: D105
        return len(self.data)

    def __getitem__(self, idx: int) -> T:
        """Returns the item at a given index.

        Return: Elements of the list like object.

        Note:
            Datapoints should be compatible with DataLoader. General we work with numerics and matrixes
            (e.g tuples, dicts, int, float, numpy, torch). DataLoader internally convers to torch object.
            For specific DataLoader transformation/consumption see `torch._utils.collate.default_collate`.
        """
        return cast(T, self.data[idx])
