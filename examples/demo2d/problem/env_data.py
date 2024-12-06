"""Handle the collection of data."""

# %%
from pathlib import Path

import numpy as np
import pandas as pd


def collect_data() -> pd.DataFrame:
    """Returns a dataframe of the env data."""
    current_dir = Path(__file__).parent
    numpy = np.load(current_dir / "data/environment_distribution.npy")
    return pd.DataFrame(numpy, columns=["x1", "x2"])
