# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**axtreme** is a Python library extending Ax (Facebook's Adaptive Experimentation) and BoTorch (Bayesian Optimization in PyTorch) for design of experiments, active learning, and extreme response analysis. It targets reliability engineering scenarios like estimating 50-year storm loads from surrogate models.

## Build & Development Commands

```bash
uv sync                              # Install all dependencies (dev included)
uv sync --extra cuda                 # With CUDA support
uv sync --extra examples             # With example dependencies (openturns, numba)
uv run pre-commit install            # Setup pre-commit hooks

# Testing
uv run pytest                        # Run all tests
uv run pytest tests/path/test_file.py::test_name  # Single test
uv run pytest -m "not system"        # Skip long-running system tests
uv run pytest --cov                  # With coverage

# Linting & Formatting
uv run ruff format                   # Format code
uv run ruff check --fix              # Lint with auto-fix
uv run pyright                       # Type checking (primary)
uv run mypy                          # Type checking (secondary)
uv run pre-commit run --all-files    # Run all checks (ruff, pyright, mypy)
```

## Architecture

### Core Protocols (structural subtyping via `typing.Protocol`)

The codebase is designed around three key protocols that define the extension points:

- **`Simulator`** (`simulator/base.py`): `__call__(x: ndarray[n_points, n_dims], n_simulations_per_point) -> ndarray[n_points, n_sims, n_outputs]` -- wraps any simulation model.
- **`QoIEstimator`** (`qoi/qoi_estimator.py`): `__call__(model: Model) -> Tensor[n_estimates]` -- estimates a scalar quantity of interest from a BoTorch surrogate model. Has `mean()` and `var()` methods for aggregation (overridable for special samplers like UT).
- **`PosteriorSampler`** (`sampling/base.py`): `__call__(posterior: GPyTorchPosterior) -> Tensor` -- draws samples from GP posteriors with different strategies.

### Bayesian Optimization Loop (how modules connect)

1. **Experiment setup** (`experiment.py`): Create Ax Experiment, initialize with Sobol points
2. **Simulation** (`evaluation.py`): `EvaluationFunction` wraps a `Simulator`, runs it, fits a distribution (Gumbel) to outputs -> `SimulationPointResults`
3. **Ax integration** (`runner.py` + `metrics.py`): `LocalMetadataRunner` executes evaluation, stores results in trial metadata; `LocalMetadataMetric` fetches them back
4. **Surrogate model**: Ax/BoTorch fits a GP to collected data
5. **QoI estimation** (`qoi/`): Either `GPBruteForce` (full-period simulation) or `MarginalCDFExtrapolation` (CDF^N extrapolation) estimates the extreme response quantity
6. **Acquisition** (`acquisition/qoi_look_ahead.py`): `QoILookAhead` uses fantasy models to select the next point that most reduces QoI variance
7. Repeat from step 2

### Key Module Purposes

- **`qoi/`**: Two strategies for extreme value estimation. `GPBruteForce` simulates all timesteps per period. `MarginalCDFExtrapolation` approximates via single-timestep CDF raised to power N -- much faster for large N.
- **`sampling/`**: `MeanSampler` (no uncertainty), `IndependentMCSampler` (diagonalizes cross-point covariance), `NormalIndependentSampler`, `UTSampler` (unscented transform, needs custom mean/var).
- **`distributions/`**: `ApproximateMixture` (conservative tail extrapolation for safety-critical use), `icdf` (inverse CDF via root-finding for distributions without analytic inverse).
- **`data/`**: PyTorch Dataset/Sampler wrappers including `NumpyFileDataset` (memory-mapped), importance sampling support, custom batch samplers.
- **`utils/transforms.py`**: Converts Ax transforms to BoTorch space so models can operate in problem space rather than Ax's normalized space.
- **`eval/`**: `QoIJob`/`QoIJobResult` for organizing and serializing QoI evaluation runs.

### Tensor Dimension Conventions (BoTorch notation)

- `*b`: batch dimensions (arbitrary)
- `n`: number of input points
- `m`: output dimensionality
- GP posterior shape: `(*b, n, m)`

## Code Style

- **Formatter/Linter**: Ruff (configured in `ruff.toml`). Selects ALL rules then ignores specific ones.
- **Line length**: 120 characters
- **Docstrings**: Follow Google style.
- **Type hints**: Required on all functions. Pyright in basic mode with strict-ish overrides. Stubs in `stubs/` directory for untyped third-party libraries.
- **Imports**: Absolute only (no relative imports). Grouped: stdlib, third-party, local.
- **Test markers**: `integration`, `external`, `system` (very long), `non_deterministic`. Tests mirror `src/` structure.
- **TODOs**: Use `# @TODO: Description. AUTHOR, YYYY-MM-DD` format.

## Important Constraints

- **numpy < 2.0**: Pinned for compatibility with the scipy/torch/botorch ecosystem.
- **ax-platform == 0.3.7**: Pinned exact version; Ax APIs can change significantly between versions.
- **Single output only**: Multi-output GP support is not yet implemented across the QoI pipeline.
