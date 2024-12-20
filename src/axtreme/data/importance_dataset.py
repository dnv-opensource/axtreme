"""Dataset that return importance sample information in the form (data, importance_weight).

Dev details:
    Datasets can return a variety of things when ``__get_item__`` is called, for example:
    - `Tuples <https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#getitem>`_
    - `Dicts <https://pytorch.org/tutorials/beginner/data_loading_tutorial.html#dataset-class>`_

    The dataloader will respect objects like ``dict`` and ``tuple``, and will convert float/int/list/numpy content into
    tensors as it is assumed to be data.
    - By default done by ``collate_fn`` arg of Dataloader.
    - For defaults see ``torch.utils.data._utils.collate.default_collate``

    While this is straight forward to implement, typing can be a challenge (``torch.utils.data.Dataset`` provides some
    guidances, but appears they found it challenging too).

Todo:
    - Revisit the typing and the implications of covariate/contravariant etc.
"""

# pyright: reportUnnecessaryTypeIgnoreComment=false
# %%
from typing import Any, TypeVar

import numpy as np
import torch
from torch.utils.data import Dataset, StackDataset
from torch.utils.data.dataset import T_co, T_tuple

T_np_interface = TypeVar("T_np_interface", np.ndarray[tuple[int, ...], Any], torch.Tensor)


class ImportanceIndexWrapper(Dataset[tuple[T_np_interface, float]]):
    """Wraps an existing dataset, returning a column/index of the item as the importance weight."""

    def __init__(self, dataset: Dataset[T_np_interface], importance_idx: int) -> None:
        """Wrap an existing dataset, returning part of the item as the importance weight.

        Args:
            dataset: A dataset, where one of the columns should be used as an importance weight.
                - Note: Currently only datasets that return numpy or torch tensors are supported
            importance_idx: the column index containing the importance weights.

        Todo:
            - Generalise this to deal with other types of dataset output (list, numpy etc).
        """
        self.dataset: Dataset[T_np_interface] = dataset
        self.importance_idx: int = importance_idx

        datapoint = self.dataset[0]

        self.mask = torch.ones(datapoint.shape, dtype=torch.bool)
        self.mask[importance_idx] = False

    def __len__(self) -> int:  # noqa: D105
        return len(self.dataset)  # type: ignore[arg-type]

    def __getitem__(self, index: int) -> tuple[T_np_interface, float]:
        """Gets the data and importance weight at the requested index in the dataset.

        Args:
            index: what index in the dataset should be returned

        Returns:
            A tuple where:
            - Element 0: (tensor) the datapoint
            - Element 1: Float representing the importance weight
        """
        datapoint = self.dataset[index]

        # datapoint with importance column, importance weight
        return datapoint[self.mask], float(datapoint[self.importance_idx])  # pyright:  ignore[reportReturnType]


class ImportanceAddedWrapper(StackDataset[T_tuple]):  # pyright: ignore[reportMissingTypeArgument]
    "Thin wrapper makes the method for creating the dataset more explicit, and ensure order and type of output."

    def __init__(self, data_dataset: Dataset[T_co], importance_dataset: Dataset[T_co]) -> None:
        """Ensures the StackDataset is created as a tuple regardless of the way args are passed."""
        super().__init__(data_dataset, importance_dataset)
