import torch
from torch.distributions.distribution import Distribution

class TransformedDistribution(Distribution):
    def sample(self, sample_shape: torch.Size = ...) -> torch.Tensor: ...
    def rsample(self, sample_shape: torch.Size = ...) -> torch.Tensor: ...
