# %%  # noqa: D100


import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from .axtreme_case import sample_seastates


def generate_data(n_years_in_period: int, n_sea_states_in_year: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Genarates environment data.

    Args:
        n_years_in_period: number of years considered
        n_sea_states_in_year: number of sea states in one year

    Returns:
        significant wave height and peak wave period

    """
    weib_prms = (1.2550, 0.5026, 1.9536)
    lognorm_prms = (1.557, 0.403, 0.408, 0.005, 0.137, 0.454)

    n_ss = n_sea_states_in_year * n_years_in_period
    return sample_seastates(n_ss, weib_prms, lognorm_prms, hslim=7.5)


# %%
if __name__ == "__main__":
    N_y = 1000
    Hs, Tp = generate_data(n_years_in_period=N_y, n_sea_states_in_year=2922)

    plt.figure()  # type: ignore  # noqa: PGH003
    plt.plot(Hs, Tp, ".")  # type: ignore  # noqa: PGH003

    # Save the data as a .npy file
    np.save(f"long_term_distribution_{N_y}_years.npy", np.column_stack((Hs, Tp)))
# %%
