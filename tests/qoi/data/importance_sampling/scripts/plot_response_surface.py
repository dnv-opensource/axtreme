"""Plot the response surface of the ImportanceSamplingTestSimulator."""

# %%
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm
from simulator import _true_underlying_func

# %%
# Create a grid over the input space
n = 200
x1 = np.linspace(-1.0, 3.0, n)
x2 = np.linspace(-1.0, 3.0, n)
X1, X2 = np.meshgrid(x1, x2)

# Evaluate the true underlying function on the grid
grid_points = torch.tensor(np.column_stack([X1.ravel(), X2.ravel()]), dtype=torch.float64)
params = _true_underlying_func(grid_points)
loc = params[:, 0].numpy().reshape(n, n)
scale = params[:, 1].numpy().reshape(n, n)

# %%
# --- Plot ---
fig = plt.figure(figsize=(16, 5))

# 1) 3D surface of loc(x)
ax1 = fig.add_subplot(131, projection="3d")
ax1.plot_surface(X1, X2, loc, cmap=cm.viridis, edgecolor="none", alpha=0.9)
ax1.set_xlabel("x1")
ax1.set_ylabel("x2")
ax1.set_zlabel("loc(x)")
ax1.set_title("Gumbel loc (3D surface)")

# 2) Contour of loc(x)
ax2 = fig.add_subplot(132)
cf = ax2.contourf(X1, X2, loc, levels=40, cmap=cm.viridis)
ax2.set_xlabel("x1")
ax2.set_ylabel("x2")
ax2.set_title("Gumbel loc (contour)")
fig.colorbar(cf, ax=ax2)

# 3) Contour of scale(x) — constant but shown for completeness
ax3 = fig.add_subplot(133)
cf2 = ax3.contourf(X1, X2, scale, levels=40, cmap=cm.plasma)
ax3.set_xlabel("x1")
ax3.set_ylabel("x2")
ax3.set_title("Gumbel scale (contour)")
fig.colorbar(cf2, ax=ax3)

fig.suptitle("Response surface: Gumbel(loc(x), scale(x))", fontsize=14, y=1.02)
fig.tight_layout()

# %%
