"""Additional helpers."""

# %%
import json
from pathlib import Path
from typing import Any


def append_to_json(obj: Any, output_file: Path) -> None:  # noqa: ANN401
    """Appends a json object to a file containing a list of objects.

    object: Json serialisable object to append.
    output_file: The file to append results to. If does not exist, it will be created.
    """
    # Initialise the file if it doesn't exist
    if not output_file.exists():
        # Create parent dir if required
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open("w") as fp:
            json.dump([], fp)

    with output_file.open("r") as fp:
        existing_results = json.load(fp)

    existing_results.append(obj)

    with output_file.open("w") as fp:
        json.dump(existing_results, fp)
