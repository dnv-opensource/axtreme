"""This script give recommendations on how to incorporate environment data.

Incorporating environment data into QoI calculation is a common task. The `axtreme` package assumes users have samples
of their environmental data that they need to use. Typcially, these samples are only ever used in the QoI method, so
this is the only thing they are required to be compatable with (e.g if you make a custom QoI you can include env data
however you would like). The `axtreme` QoI methods use `torch.Dataloader` as a convenient was to use environment data.
The following provides a quick intro of how to set them up and confirgure them for different tasks.

The content is organised in the following sections:
- Basics
- Typical set up.
- Data vs. Distibution
"""

# %%
import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import DataLoader, Dataset, RandomSampler

from axtreme.data import BatchInvariantSampler2d, SizableSequentialSampler

# %%
"""Basics: torch components.
The following show the most basic verion of the torch components we use in axtreme.
NOTE: We make use of a more advanced set up detailed in the next section.
"""


# Create a Dataset to represent your data.
class MinimalDataset(Dataset[torch.Tensor]):
    """See axtreme.data for more detailed examples."""

    def __init__(self, data: NDArray[np.float64 | np.int32]) -> None:  # noqa: D107
        self.data = data

    def __len__(self) -> int:  # noqa: D105
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:  # noqa: D105
        return torch.from_numpy(self.data[idx, :])


# Our fake dataset has 8 rows of data, and 2 columns
data = np.arange(16).reshape(-1, 2)
dataset = MinimalDataset(data=data)

dataloader = DataLoader(dataset, batch_size=5, num_workers=0)
for d in dataloader:
    print(d.shape)

# %%
"""Typical set up.

This set up is more advance and is typically use in GPBruteForce QoI. The key difference is that here we are interested
in iterating through as "2d" dataset of points (e.g the datapoints are arrange into rows and columns with the overall
shape (n_rows, n_cols, data_dim)).

The following details the more advanced set up we use to achieve:
- Returning batches of the "2d" arrangement.
- Ensuring batchs are batch_size invariant.



Specific components used to achieve this:
- `DataSet` store the data and control access to it.
- A `Sampler` controls the amount of data that will be used (and which points are selected).
    - Sequential samplers are currently prefered because:
        - Repeat iteration of the DataLoader will produce the same result.
        - If the DataSet uses files rather than all being stored in memory, sequential reads can easily be cached.
- A `BatchSampler` can be used to control the shape in which the data is consumed.
- A `DataLoader` is used to orchestrated the above components.
    - It also provides effecient loading and transfer of data through `num_workers` and `pin_memory`

`axtreme` objects that use the dataloader simply consume all its output (similar to what happens in standard machine
learning)
"""

# the number of rows in the '2d' dataset
n_periods = 2
# the number of columns in the `2d` dataset (e.g the number of samples in each period)
period_len = 6


# Define the size of the data our system will see.
# This determine the total amount of data the following step must consume.
# This can be larger that the amount of data truely underlying
sampler = SizableSequentialSampler(dataset, num_samples=n_periods * period_len)

# Define the shape of the batches used
batch_invariant_sampler = BatchInvariantSampler2d(sampler=sampler, batch_shape=torch.Size([n_periods, 4]))

# Show how the output will be arranged (data is referenced by its index here)
batches = [torch.tensor(b) for b in batch_invariant_sampler]
print("batches (idx):\n", batches)
# The dataset this effectively means you are using
print("\nconcatenated:\n", torch.concat(batches, dim=-1))

# Put into a dataloader to orchestrate loading the data.
print("\n\n batches from dataloader")
dataloader = DataLoader(dataset, batch_sampler=batch_invariant_sampler)
for idx, d in enumerate(dataloader):
    print(f"batch {idx} shape {d.shape}")

for idx, d in enumerate(dataloader):
    print(f"batch {idx}:  {d}")
# %%
"""
Data vs Distibution:
The first distinction/decision is if environment data should be thought of as a static dataset, or a distibution we are
sampling from.

    Data:
    - Like in classic machine learning, this is a static dataset, and we typically want to use all the data available.
    - We can stop early or over sample the data if needed, but typically we would want to use all the data so we can
      extract as much information as possible from it. e.g we should use each datapoint before reusing points.

    The key distinction is sampling WITHOUT replacement.

    Distibution:
    - the data we have is big enough that randomly sampling from it is compariable to sampling from a distibution.
    - The key distinct is sampling WITH replacement

Both of these can be easily achieved by addjusting the sampler
"""
# %%
"""Treat the underlying as a Distibution"""
generator = torch.Generator()
_ = generator.manual_seed(7)
replacement_sampler = RandomSampler(
    dataset,
    num_samples=n_periods * period_len,
    generator=generator,
    replacement=True,
)
batch_invariant_sampler = BatchInvariantSampler2d(
    sampler=replacement_sampler,
    batch_shape=torch.Size([n_periods, 4]),
)
batches = [torch.tensor(b) for b in batch_invariant_sampler]
print("batches idx:\n", batches)
# The dataset this effectively means you are using
print("\nconcatenated idxs:\n", torch.concat(batches, dim=-1))

print("\n\n batches from dataloader")
dataloader = DataLoader(dataset, batch_sampler=batch_invariant_sampler)
for d in dataloader:
    print(d.shape)

# NOTE: Each time your run the dataloader (even with the same fixed starting seed - get different sampler)
assert not (torch.concat(list(dataloader), -2) == torch.concat(list(dataloader), -2)).all()
