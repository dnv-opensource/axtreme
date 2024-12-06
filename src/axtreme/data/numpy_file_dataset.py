"""Datasets to help work with numpy files (.npy)."""

# %%
import functools
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset


@functools.lru_cache(maxsize=1)
def collect_np_file(file_path: str | Path) -> NDArray[np.float64]:  # noqa: D103
    return np.load(file_path)


class NumpyFileDataset(Dataset[torch.Tensor]):
    """Helper to work with directories of .npy data.

    Note:
        - Highly recommened to use an in memory dataset if possible. This is typically a bottleneck.
        - Using with a sequential sampler will be significantly faster because this performs rudimental cacheing.
            - Random sampling will require from disk read for EVERY datapoint. Suggest randomise the save files.

    Assumes:
        - Each row is a data point
        - Each file has the same number of datapoints within it.

    Dev:
        - Based on example here: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

    Todo:
        - This is slow compared to reading from memory. 100k dataset (3ms for memory, 20s with this dataloader)
            - This is because EVERY SINGLE datapoint requireds a new read from disk.
            - TODO: Consider different file types that might be more appropriate (row specfic access)
                - HDF5?
            - TODO: consider intergration with an imporance weight
                - Look at the example with images, can return many things
                - npz allows you to combine multiple arrays, could have soem logic if other array none, no imporance.
    Answered:
        - Is Sampler a more approapriate way of framing this?
            - No. The samples class just take an existing dataset and shuffles it. Like shuffle in dataloader.
    """

    def __init__(self, root_dir: str | Path) -> None:
        """Initialise the Dataset.

        Note:
            Data should be loaded lazily (in `__getitem__`, not here)

        Arguments:
            root_dir (string): Directory with .npy files
        """
        self.root_dir = root_dir
        self.paths = list(Path(root_dir).glob("*.npy"))
        self.row_per_file = np.load(self.paths[0]).shape[0]

    def __len__(self) -> int:  # noqa: D105
        # This doesn't do anything to control when samplers will stop sampling. Just tells the range of valid indexs
        # RandomSampler might sample many more points than is in this dataset (even if replace = False)
        # Sequential will just iterate though once

        # Basically: When stop drawing data is based on the logic in the sampler.
        return self.row_per_file * len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """In this function you want to get only the index of data required.

        Args:
            idx: what index in the dataset should be returned

        Returns:
            A tensor representing a single datapoint.

        Note:
            See the example linked in class docs, can return whatever type of object would like.
            This could be useful for importance sampling.
        """
        file_idx = idx // self.row_per_file
        row_idx = idx % self.row_per_file

        file = collect_np_file(self.paths[file_idx])
        sample = file[row_idx, :]
        # NOTE: this should NOT be on GPU. That is done lazily as needed in the training loop.
        # If put on GPU now Dataloader works can't build an effecient GPU queue.
        return torch.from_numpy(sample)
