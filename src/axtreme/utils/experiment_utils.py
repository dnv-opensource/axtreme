"""Helpers for working with the Ax Experiment class."""

import pandas as pd
from ax import Experiment


def input_out_df_from_experiment(exp: Experiment) -> pd.DataFrame:
    """Shows the x and y data in one dataframe."""
    df_y = exp.fetch_data().df
    df_x = pd.DataFrame([{**trial.arm.parameters, "arm_name": trial.arm.name} for trial in exp.trials.values()])  # pyright: ignore[reportAttributeAccessIssue]
    return df_y.merge(df_x, on="arm_name", how="inner")
