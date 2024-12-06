"""Classes to organize the evaluation of a QoIs."""

# pyright: reportUnnecessaryTypeIgnoreComment=false

# %%
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from botorch.models.model import Model
from dataclasses_json import config, dataclass_json

import axtreme.eval.object_logging
from axtreme.eval import utils
from axtreme.qoi import QoIEstimator


@dataclass_json
@dataclass
class QoIJobResult:
    """Dataclass to store the results of running a QoIEstimator (as produced by QoIJob).

    Args:
        mean: the mean of the QoIEstimator results.
            Note, if UT methods are used in the QoIEstimator, this can't be calculated directory form samples
        var: variance in the QoIEstimator results.
            Note, if UT methods are used in the QoIEstimator, this can't be calculated directory form samples
        samples: Samples produced by the QoIEstimator
        tags: Label assigned to the result by the user. This information can typically be found somewhere in metadata,
            putting them here is for convience (more accessible and less noise).
        metadata: Optional indepth information about the conditions that produced these results.
    """

    # 0d tensors
    mean: torch.Tensor = field(metadata=config(encoder=torch.Tensor.tolist, decoder=torch.tensor))
    var: torch.Tensor = field(metadata=config(encoder=torch.Tensor.tolist, decoder=torch.tensor))
    # all samples
    samples: torch.Tensor = field(metadata=config(encoder=torch.Tensor.tolist, decoder=torch.tensor))

    tags: dict[str, str | float] = field(default_factory=dict)

    metadata: dict[str, Any] = field(default_factory=dict)

    # Implementation from dataclasses_json. Method stub to assist type checkers
    def to_dict(self) -> dict[str, Any]: ...  # type: ignore[empty-body]  # noqa: D102

    def from_dict(self) -> "QoIJobResult": ...  # type: ignore[empty-body]  # noqa: D102


@dataclass
class QoIJob:
    """Helper to make the interface below clear."""

    model: Model
    qoi: QoIEstimator
    name: str | None = None
    # easy access to key that will be used to search the job
    tags: dict[str, str | float] = field(default_factory=dict)
    # general metadata
    metadata: dict[str, str | float] = field(default_factory=dict)

    def __call__(self, output_file: None | Path = None) -> QoIJobResult:
        """Run the QoIJob.

        Args:
            output_file: If provided a json version of the results is appended to this file. Expects the file to be a
            list object. The file will be created if it does not exist.

        Returns:
            QoIJobResult
        """
        start_time = time.time()
        results = self.qoi(self.model)
        duration = time.time() - start_time

        mean = self.qoi.mean(results)
        var = self.qoi.var(results)
        metadata = {
            **self.metadata,
            "runtime_sec": duration,
            "qoi": axtreme.eval.object_logging.unpack_object_str_content(self.qoi, depth=2),
            "model": axtreme.eval.object_logging.unpack_object_str_content(self.model),
        }

        result = QoIJobResult(mean=mean, var=var, samples=results, tags=self.tags, metadata=metadata)

        if output_file:
            utils.append_to_json(result.to_dict(), output_file)

        return result
