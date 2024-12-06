"""Sequential sampling."""

from collections.abc import Iterator, Sized

from torch.utils.data import Sampler


class SizableSequentialSampler(Sampler[int]):
    """Samples elements sequentially, always in the same order.

    Follows the pattern in RandomSampler to allow for sampling a specific number of samplers.
    This can be smaller or larger than the amount in the dataset.
    """

    data_source: Sized  # this follow the patter in SequentialSampler

    def __init__(self, data_source: Sized, num_samples: None | int = None) -> None:
        """Create the sampler.

        Args:
            data_source (Dataset): dataset to sample from
            num_samples (int): number of samples to draw, default=`len(dataset)`.
        """
        self.data_source = data_source
        self._num_samples = num_samples

    @property
    def num_samples(self) -> int:  # noqa: D102
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self) -> Iterator[int]:  # noqa: D105
        n = len(self.data_source)

        for _ in range(self.num_samples // n):
            yield from iter(range(n))
        yield from iter(range(self.num_samples % n))

    def __len__(self) -> int:  # noqa: D105
        return self.num_samples
