# %%
"""Creates importance samples using the function in `importance_sampling.py` and performs checks.

This file will be built in the following iteration which have increased realism:
- Iter1: Goal: Demonstrate importance sampling works.
    - "Cheat" and use the true env distribution and brute force xmax results to get "perfect" importance samples.
- Iter2: Goal: Generate importance samples using real world constraints.
    - e.g use KDE to fit the environment samples.
    - Explain how to pick hyperparams (e.g threshold) without brute force "answers".

This is currently at iteration 1.
"""

# %%
import json

import numpy as np
import torch
from importance_sampling import importance_sampling_distribution_uniform_region  # type: ignore[import-not-found]
from matplotlib import pyplot as plt
from usecase.env_data import env_pdf  # type: ignore[import-not-found]

torch.set_default_dtype(torch.float64)


# %% Get the true PDF of the environment distribution.
def torch_env_pdf(x: torch.Tensor) -> torch.Tensor:
    """Converts the env_pdf to a torch function.

    Args:
        x: A tensor of shape (n, d) where n is the number of samples and d is the dimension of the input space.

    Returns:
        A tensor of shape (n,) containing the pdf values for the input samples.
    """
    x_np = x.numpy()
    pdf_values = env_pdf(x_np)
    return torch.tensor(pdf_values, dtype=torch.float64)


# %% Run the importance sampling method
importance_samples, importance_weights = importance_sampling_distribution_uniform_region(
    env_distribution_pdf=torch_env_pdf,
    region=torch.tensor([[0.1, 1], [30, 30]]),
    # lowest pdf value are that is included.
    threshold=1e-10,
    num_samples_total=150_000,
)
# %%
""" Evaluate the importance sampling regions:

For iteration 1 we cheat and visually check the important `xmax` values are within our distribution.

"""
print("Number of importance samples: ", importance_samples.shape[0])
_ = plt.scatter(importance_samples[:, 0], importance_samples[:, 1], s=1, c="tab:blue", label="Importance samples")

with open("results/brute_force/29220_period_length.json") as f:  # noqa: PTH123
    brute_force = json.load(f)
xmax = np.array(brute_force["env_data"])

_ = plt.scatter(xmax[:, 0], xmax[:, 1], s=1, c="tab:orange", label="Brute force samples")

# %% plot the importance weights
_ = plt.scatter(
    importance_samples[:, 0], importance_samples[:, 1], s=1, c=importance_weights, label="Importance samples"
)

# %% Save the result if we are happy with them.
torch.save(importance_samples, "results/importance_sampling/importance_samples.pt")
torch.save(importance_weights, "results/importance_sampling/importance_weights.pt")
# %%
