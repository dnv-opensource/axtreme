"""helpers for working with distribution."""

import torch
from torch.distributions import Distribution, MixtureSameFamily


def index_batch_dist(dist: Distribution, index: tuple[slice | int, ...]) -> Distribution:
    """Applies indexing to a MixtureSameFamily object along the batch dimension.

    Args:
        dist: The distribution to be indexed (only along the batch dimensions)
        index: The index/slice to be applied. e.g `(slice(1,4), Ellipsis)` is equivalent to [1:4,...].
            Slices larger than the batch dimension will cause index error.

    Returns:
        A "veiw" of the underling distribution. This is a new object, but we call it a veiw as it is built on a view of
        the underling data.

    Note:
        The returned distribution is built on a view of the underling data. The follows the behaviour of slicing tensors
        in pytorch as detailed `here <https://discuss.pytorch.org/t/does-indexing-a-tensor-return-a-copy-of-it/164905>`_.
        As such gradients etc are connected.
    """
    # Mixture distribution require special handling because they have multiple distributions internally.
    if isinstance(dist, MixtureSameFamily):
        return index_batch_mixture_dist(dist, index)

    # NOTE: unclear if all dists have a populated arg_constraints. If not, this will fail and we will be alerted to
    # the issues
    params = {param: getattr(dist, param)[index] for param in dist.arg_constraints}
    return dist.__class__(**params)


def index_batch_mixture_dist(dist: MixtureSameFamily, index: tuple[slice | int, ...]) -> MixtureSameFamily:
    """Applies indexing to a MixtureSameFamily object along the batch dimension.

    Args:
        dist: The distribution to be indexed (only along the batch dimensions)
        index: The index/slice to be applied. e.g `(slice(1,4), Ellipsis)` is equivalent to [1:4,...].
            Slices larger than the batch dimension will cause index error.

    Returns:
        A "veiw" of the underling distribution. This is a new object, but we call it a veiw as it is built on a view of
        the underling data.

    Note:
        The returned distribution is built on a view of the underling data. The follows the behaviour of slicing tensors
        in pytorch as detailed `here <https://discuss.pytorch.org/t/does-indexing-a-tensor-return-a-copy-of-it/164905>`_.
        As such gradients etc are connected.
    """
    if len(index) > len(dist.batch_shape):
        raise IndexError(f"too many indices for tensor of dimension {dist.batch_shape}")

    # NOTE: unclear if all dists have a populated arg_constraints. If not, this will fail and we will be alerted to
    # the issues

    comp_slice = {
        param: getattr(dist.component_distribution, param)[index]
        for param in dist.component_distribution.arg_constraints
    }
    # both probs and logits are automatically populated as soon as one is provided. We an pick either.
    mix_slice = {"probs": dist.mixture_distribution.probs[index]}

    comp_new = dist.component_distribution.__class__(**comp_slice)
    mix_new = dist.mixture_distribution.__class__(**mix_slice)

    return dist.__class__(mix_new, comp_new)


def dist_dtype(dist: Distribution) -> torch.dtype:
    """Return the dtype the distribution calculates values in.

    Parameters may be of different tpyes. It appears the distribution defaults to the largest dtype.

    Args:
        dist: the distribution to find the dtype of.

    Returns:
        dtype the ditribution returns result in.
    """
    # All distribution support this attribute. This is the easiest way to identify the dtype.
    return dist.mean.dtype
