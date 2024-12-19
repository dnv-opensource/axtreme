import pytest
import torch
from torch.distributions import Categorical, Gumbel

from axtreme.distributions import ApproximateMixture
from axtreme.distributions.utils import (
    index_batch_dist,
    index_batch_mixture_dist,
)


class TestIndexBatchDist:
    def test_index_batch_mixture_dist_standard_indexing(self):
        """Basic indexing when batch shape is 2"""
        # Set up batched dist
        loc = torch.tensor([0.0, 1.0], dtype=torch.float64)
        shape = torch.ones_like(loc)
        dist = Gumbel(loc=loc, scale=shape)

        # create underling dist
        indexed_dist = index_batch_dist(dist, (0,))

        # check they produce the same results
        x = torch.tensor([0.5], dtype=torch.float64)
        q_batch = dist.cdf(x)
        q = indexed_dist.cdf(x)

        assert q.item() == q_batch[0].item()

    def test_index_batch_mixture_dist_slicing(self):
        """Basic indexing when batch shape is [5,2]"""
        # Set up batched dist
        loc = torch.rand([5, 2, 3], dtype=torch.float64)
        shape = torch.ones_like(loc)
        dist = Gumbel(loc=loc, scale=shape)

        # create underling dist
        index = (slice(0, 2), 0)
        indexed_dist = index_batch_dist(dist, index)

        # check they produce the same results
        x = torch.tensor([0.5], dtype=torch.float64)
        q_batch = dist.cdf(x)
        q = indexed_dist.cdf(x)

        torch.testing.assert_close(q, q_batch[index], rtol=0, atol=1e-10)


class TestIndexBatchMixtureDist:
    def test_index_batch_mixture_dist_standard_indexing(self):
        """Basic indexing when batch shape is 2"""
        # Set up batched dist
        loc = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.float64)
        shape = torch.ones_like(loc)
        comp = Gumbel(loc=loc, scale=shape)
        mix = Categorical(torch.ones_like(loc, dtype=torch.float64))
        dist = ApproximateMixture(mix, comp)

        # create underling dist
        indexed_dist = index_batch_mixture_dist(dist, (0,))

        # check they produce the same results
        x = torch.tensor([0.5], dtype=torch.float64)
        q_batch = dist.cdf(x)
        q = indexed_dist.cdf(x)

        assert q.item() == q_batch[0].item()

    def test_index_batch_mixture_dist_slicing(self):
        """Basic indexing when batch shape is [5,2]"""
        # Set up batched dist
        loc = torch.rand([5, 2, 3], dtype=torch.float64)
        shape = torch.ones_like(loc)
        comp = Gumbel(loc=loc, scale=shape)
        mix = Categorical(torch.ones_like(loc, dtype=torch.float64))
        dist = ApproximateMixture(mix, comp)

        # create underling dist
        index = (slice(0, 2), 0)
        indexed_dist = index_batch_mixture_dist(dist, index)

        # check they produce the same results
        x = torch.tensor([0.5], dtype=torch.float64)
        q_batch = dist.cdf(x)
        q = indexed_dist.cdf(x)

        torch.testing.assert_close(q, q_batch[index], rtol=0, atol=1e-10)

    def test_index_batch_mixture_dist_index_that_would_effect_component_distributions(self):
        """Batch same is (2,), indexing (2,1) should raise an error."""
        # Set up batched dist
        loc = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.float64)
        shape = torch.ones_like(loc)
        comp = Gumbel(loc=loc, scale=shape)
        mix = Categorical(torch.ones_like(loc, dtype=torch.float64))
        dist = ApproximateMixture(mix, comp)

        # create underling dist
        with pytest.raises(IndexError, match=".*too many indice.*"):
            _ = index_batch_mixture_dist(dist, (2, 1))
