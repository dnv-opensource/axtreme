"""Dataset that return importance sample information."""
# pyright: reportUnnecessaryTypeIgnoreComment=false

# @TODO: The Importance...Wrapper classes break compatibility with their base class,
#        as they return a tuple instead of the expected data type.
#        On thing is that it feels a bit "dirty" from an architectural point of view.
#        More important would be if the implementation causes problems when using
#        the Importance...Wrapper Dataset classes with PyTorch's DataLoader,
#        as I suspect this expects `__get_item__()` to retrun not a tuple, but a single object (of type `T_co`).
#        To suppress pyright errors, I for now deactivated the corresponding error codes for this module (see below).
#        However, maybe there is a cleaner implementation to add weights to a dataset? @swinter @KristofferSkare
#        (_CoPilot suggestion_, not checked):
#        "One idea would be to use a `collate_fn` in the DataLoader, which could add the weights to the data."
#        @ClaasRostock, 2024-08-30
# pyright: reportIndexIssue=false
# pyright: reportReturnType=false
# pyright: reportIncompatibleMethodOverride=false
# mypy: disable-error-code="override, index, return-value"

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co


class ImportanceIndexWrapper(Dataset[T_co]):
    """Wraps an existing dataset, returning part of the item as the importance weight."""

    def __init__(self, dataset: Dataset[T_co], importance_idx: int) -> None:
        """Wrap an existing dataset, returning part of the item as the importance weight.

        Args:
            dataset: A dataset, where one of the columns should be used as an importance weight.
                - Note: Currently only datasets that return numpy or torch tensors are supported
            importance_idx: the column index containing the importance weights.

        Todo:
            - Generalise this to deal with other types of dataset output (list, numpy etc).
        """
        self.dataset: Dataset[T_co] = dataset
        self.importance_idx: int = importance_idx

        datapoint = self.dataset[0]

        # Might be more pythonic to not check up front, but I think its a better user experience
        # These have the same indexing interface so are easy to support
        if not (isinstance(datapoint, np.ndarray | torch.Tensor)):
            msg = (
                "ImportanceIndexWrapper only support datasets where the underlying data is"
                f"torch.Tensor or NDArray[np.float64]. Instead got {type(datapoint)}"
            )
            raise TypeError(msg)

        self.mask = torch.ones(datapoint.shape, dtype=torch.bool)
        self.mask[importance_idx] = False

    def __len__(self) -> int:  # noqa: D105
        return len(self.dataset)  # type: ignore[arg-type]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
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
        return datapoint[self.mask], datapoint[self.importance_idx]


class ImportanceAddedWrapper(Dataset[T_co]):
    """Combine existing dataset with one containing the related importance weights."""

    def __init__(self, data_dataset: Dataset[T_co], importance_dataset: Dataset[T_co]) -> None:
        """Combines one Dataset of data, with a Dataset of importance weights.

        Args:
            data_dataset: Dataset containing input data
            importance_dataset: Contains importance weights.
                - `__get_item__()` should return floats.

        Note:
            Both datasets need to share the same index.
            E.g. `importance_datasets[idx]` provides the importance weight for the data `data_dataset[idx]`.
        """
        if len(data_dataset) != len(importance_dataset):  # type: ignore[arg-type]
            msg = (
                "Expected data_dataset and importance_dataset to have sample length."
                f" Instead got {len(data_dataset)=}, and {len(importance_dataset)}."  # type: ignore[arg-type]
            )
            raise ValueError(msg)

        self.data_dataset = data_dataset
        self.importance_dataset = importance_dataset

    def __len__(self) -> int:  # noqa: D105
        return len(self.data_dataset)  # type: ignore[arg-type]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Gets the data and importance weight at the requested index in the dataset."""
        # datapoint with importance column, importance weight
        return self.data_dataset[index], self.importance_dataset[index]
