# %%  # noqa: D100

import matplotlib.pyplot as plt
import numpy as np

from .axtreme_case import sample_seastates

# %%
# example use/testing
weib_prms = (1.2550, 0.5026, 1.9536)
lognorm_prms = (1.557, 0.403, 0.408, 0.005, 0.137, 0.454)
h = 110  # water depth 110 m


N_y = 10**4
n_ss = 2922 * N_y
Hs, Tp = sample_seastates(n_ss, weib_prms, lognorm_prms, hslim=7.5)
plt.figure()  # type: ignore  # noqa: PGH003
plt.plot(Hs, Tp, ".")  # type: ignore  # noqa: PGH003


# %%
# Save the data as a .npy file
np.save("long_term_distribution.npy", np.column_stack((Hs, Tp)))
# %%
