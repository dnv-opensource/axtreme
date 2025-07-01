"""
Plan:
    - Unit test:
        - _parameter_estimates: input and output weights are attached to the same samples
    - Integration test:
        - successful run of minimal version of the qoi using a deterministic model with importance sampling
"""

# %%
import torch
from botorch.models.deterministic import GenericDeterministicModel

from axtreme.qoi.marginal_cdf_extrapolation import MarginalCDFExtrapolation


def test_parameter_estimates_consistency_of_weights(gp_passthrough_1p: GenericDeterministicModel):
    importance_samples = torch.Tensor([1, 2, 3, 4])
    importance_weights = torch.Tensor([0.1, 0.2, 0.3, 0.4])

    samples = [importance_samples, importance_weights]

    env_sample = torch.tensor([[[0], [1], [2]]], dtype=torch.float64)
    qoi_estimator = MarginalCDFExtrapolation(env_iterable=env_sample, period_len=3)

    posterior_samples, importance_weights_qoi = qoi_estimator._parameter_estimates(gp_passthrough_1p)

    print(importance_samples, importance_weights)
    print(posterior_samples, importance_weights_qoi)


# %%
if __name__ == "__main__":
    import sys
    from pathlib import Path

    root_dir = Path("../../")
    sys.path.append(str(root_dir))
    # from conftest import gp_passthrough_1p

    test_parameter_estimates_consistency_of_weights(gp_passthrough_1p)

# %%
