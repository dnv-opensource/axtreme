"""A copy of pytorch's RandomSampler that uses the same random sample for each iteration.

This is basically an identical copy, except the Generator has been swapped for a seed, which creates the same generator
each time the samples are iterated through (e.g the DataLoader is run to completion).
"""

from collections.abc import Iterator, Sized

import torch
from torch.utils.data import Sampler


class FixedRandomSampler(Sampler[int]):
    r"""Samples elements randomly.

    This sampler differs from `torch`'s `RandomSampler` as it will return the same random sample each time it is
    iterated in full. If without replacement, then sample from a shuffled dataset. If with replacement, then user can
    specify :attr:`num_samples` to draw.
    """

    data_source: Sized
    replacement: bool

    def __init__(
        self, data_source: Sized, num_samples: int | None = None, seed: int | None = None, *, replacement: bool = False
    ) -> None:
        """Initalise the sampler.

        Args:
            data_source: dataset to sample from
            replacement: samples are drawn on-demand with replacement if ``True``, default=``False``
            num_samples: number of samples to draw, default=`len(dataset)`.
            seed: seed for the random number generator, if `None` one will be allocated randomly.
        """
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples
        self.seed = seed or int(torch.empty((), dtype=torch.int64).random_().item())

        if not isinstance(self.replacement, bool):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise TypeError(f"replacement should be a boolean value, but got replacement={self.replacement}")

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:  # pyright: ignore[reportUnnecessaryIsInstance]
            raise ValueError(f"num_samples should be a positive integer value, but got num_samples={self.num_samples}")

    @property
    def num_samples(self) -> int:  # noqa: D102
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self) -> Iterator[int]:  # noqa: D105
        generator = torch.Generator()
        _ = generator.manual_seed(self.seed)

        n = len(self.data_source)

        if self.replacement:
            for _ in range(self.num_samples // 32):
                yield from torch.randint(high=n, size=(32,), dtype=torch.int64, generator=generator).tolist()
            yield from torch.randint(
                high=n, size=(self.num_samples % 32,), dtype=torch.int64, generator=generator
            ).tolist()
        else:
            for _ in range(self.num_samples // n):
                yield from torch.randperm(n, generator=generator).tolist()
            yield from torch.randperm(n, generator=generator).tolist()[: self.num_samples % n]

    def __len__(self) -> int:  # noqa: D105
        return self.num_samples
