"""Sequential sampling."""

from collections.abc import Iterator, Sized

from torch.utils.data import Sampler


class SizableSequentialSampler(Sampler[int]):
    """Samples elements sequentially, always in the same order.

    Follows the pattern in RandomSampler to allow for sampling a specific number of samplers.
    This can be smaller or larger than the amount in the dataset.

    Args:
        data_source: dataset to sample from.
        num_samples: number of samples to draw per iteration, default=`len(dataset)`.
        stateful: if True, each call to ``__iter__`` continues from where the previous one ended (wrapping around).
            If False (default), every iteration starts from index 0.
    """

    data_source: Sized  # this follow the patter in SequentialSampler

    def __init__(self, data_source: Sized, num_samples: None | int = None, *, stateful: bool = False) -> None:
        """Create the sampler.

        Args:
            data_source: dataset to sample from.
            num_samples: number of samples to draw per iteration, default=`len(dataset)`.
            stateful: if True, each iteration continues from where the last one ended.
                If False (default), every iteration starts from index 0.
        """
        self.data_source = data_source
        self._num_samples = num_samples
        self.stateful = stateful
        self._offset = 0

    @property
    def num_samples(self) -> int:  # noqa: D102
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self) -> Iterator[int]:  # noqa: D105
        n = len(self.data_source)
        start = self._offset if self.stateful else 0

        for i in range(self.num_samples):
            yield (start + i) % n

        if self.stateful:
            self._offset = (start + self.num_samples) % n

    def __len__(self) -> int:  # noqa: D105
        return self.num_samples
