"""Script to visualize logged results from the DoE process."""

# %%
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from cmcrameri import cm as cmc
from IPython.display import display

_: Any


# %%
def json_to_dataframe(json_path: str) -> pd.DataFrame:
    """Read DoE log JSON file.

    Args:
        json_path: Path to the JSON log file.
    """
    with Path(json_path).open() as f:
        data = json.load(f)

    rows = []

    for outer_key, outer_value in data.items():
        # each outer value has exactly one inner key like "0_0"
        for inner_key, content in outer_value.items():
            row = {"entry_id": outer_key, "run_id": inner_key}

            # Extract parameters
            row.update(dict(content["parameters"].items()))

            # Extract metrics (mean and sem)
            for metric_name, metric_content in content["metrics"].items():
                for stat_name, stat_value in metric_content.items():
                    row[f"{metric_name}_{stat_name}"] = stat_value

            rows.append(row)

    df = pd.DataFrame(rows)
    return df


# %% Read DoE log file and convert to DataFrame
df = json_to_dataframe("../results/doe_log_2025_11_26_12_30.json")
display(df)

# %% Make QoI convergence plot
_ = plt.plot(df["entry_id"], df["QoIMetric_mean"], marker="o", label="QoI Mean")
_ = plt.fill_between(
    df["entry_id"],
    df["QoIMetric_mean"] - 1.96 * df["QoIMetric_sem"],
    df["QoIMetric_mean"] + 1.96 * df["QoIMetric_sem"],
    label="90% Confidence Bound",
    alpha=0.3,
)
_ = plt.xlabel("DoE Iteration")
plt.show()

# %% Plot added points
_ = plt.scatter(df["x1"], df["x2"], c=range(len(df)), cmap=cmc.batlow_r)

for i, label in enumerate(df["entry_id"]):
    _ = plt.annotate(label, (df["x1"][i], df["x2"][i]))

_ = plt.xlabel("x1")
_ = plt.ylabel("x2")
plt.show()
