import numpy as np
import torch
from torch.utils.data import DataLoader

from axtreme.data import ImportanceAddedWrapper, ImportanceIndexWrapper, MinimalDataset


def test_ImportanceAddedWrapper_numpy_input():
    dataset = MinimalDataset(
        np.array(
            [
                [0, 1, 2],
                [3, 4, 5],
            ]
        )
    )
    index_dataset = ImportanceIndexWrapper(dataset, importance_idx=2)

    # run test
    actual_data, actual_weight = index_dataset[0]

    # perform testing:
    assert list(actual_data) == [0, 1]
    assert actual_weight == 2


def test_ImportanceAddedWrapper_torch():
    dataset = MinimalDataset(
        torch.tensor(
            [
                [0, 1, 2],
                [3, 4, 5],
            ]
        )
    )
    index_dataset = ImportanceIndexWrapper(dataset, importance_idx=2)

    # run test
    actual_data, actual_weight = index_dataset[0]

    # perform testing:
    assert actual_data.tolist() == [0, 1]
    assert actual_weight == 2


def test_ImportanceAddedWrapper():
    data = np.array(
        [
            [0, 1],
            [3, 4],
        ]
    )
    data_dataset = MinimalDataset(data)
    weights = np.array([2, 5])
    importance_datasets = MinimalDataset(weights)

    dataset = ImportanceAddedWrapper(data_dataset, importance_datasets)

    actual_data, actual_weight = dataset[0]

    assert list(actual_data) == [0, 1]
    assert actual_weight == 2


##### Intergration tests
# Primarily to make sure the weight are 1d
def test_ImportanceAddedWrapper_dataloader_intergration():
    data = np.array(
        [
            [0, 1],
            [3, 4],
        ],
        dtype=np.int32,
    )
    data_dataset = MinimalDataset(data)
    weights = np.array([2, 5], dtype=np.int32)
    importance_datasets = MinimalDataset(weights)

    dataset = ImportanceAddedWrapper(data_dataset, importance_datasets)

    dl = DataLoader(dataset, shuffle=False, batch_size=2)

    actual_data, actual_weight = next(iter(dl))

    torch.testing.assert_close(
        actual_data,
        torch.tensor(
            [
                [0, 1],
                [3, 4],
            ],
            dtype=torch.int32,
        ),
    )

    torch.testing.assert_close(
        actual_weight,
        torch.tensor(
            [2, 5],
            dtype=torch.int32,
        ),
    )
