"""Interface to the information available about the environment data.

This may include:
- samples of the environment distribution.
- The distribution of the environment data (e.g. provided by an engineer).

In this usecase we assume we have access to samples of the long term. For R&D purposes the environment distribution is
also made available.
"""

# %%
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from axtreme_case import pdf_hs_tp, sample_seastates  # type: ignore[import-not-found]
from numpy.typing import NDArray


### General helpers
def collect_data() -> pd.DataFrame:
    """Returns a dataframe of the env data."""
    current_dir = Path(__file__).parent
    numpy = np.load(current_dir / "data/long_term_distribution.npy")
    return pd.DataFrame(numpy, columns=["Hs", "Tp"])


### Available for R&D purposed

_WEIB_PRMS = (1.2550, 0.5026, 1.9536)
_LOGNORMAL_PRMS = (1.557, 0.403, 0.408, 0.005, 0.137, 0.454)


def generate_data(n_years_in_period: int, n_sea_states_in_year: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Genarates environment data.

    Args:
        n_years_in_period: number of years considered
        n_sea_states_in_year: number of sea states in one year

    Returns:
        significant wave height and peak wave period

    """
    n_ss = n_sea_states_in_year * n_years_in_period
    return sample_seastates(n_ss, _WEIB_PRMS, _LOGNORMAL_PRMS, seed=1234, hslim=0)


def env_pdf(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Calculates the pdf for the environment data.

    Args:
        x: (n,2) array environment points (Hs, Tp)

    Returns:
        (n,) array of pdf values for the environment points.
    """
    hs = x[:, 0]
    tp = x[:, 1]

    pdfs = pdf_hs_tp(hs, tp, _WEIB_PRMS, _LOGNORMAL_PRMS)
    return cast("NDArray[np.float64]", pdfs)


# %%
if __name__ == "__main__":
    import numpy as np

    x = np.array([[1.0, 2], [3, 4], [5, 6]])
    pdf = env_pdf(x)
    print(pdf)

    # %%
    # Use to generate the dataset for the usecase.
    N_y = 1000
    Hs, Tp = generate_data(n_years_in_period=N_y, n_sea_states_in_year=2922)

    plt.figure()  # type: ignore  # noqa: PGH003
    plt.plot(Hs, Tp, ".")  # type: ignore  # noqa: PGH003

    # Save the data as a .npy file
    np.save("data/long_term_distribution.npy", np.column_stack((Hs, Tp)))

# %%
