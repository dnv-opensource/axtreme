# Candidate Generation Flow: `ModelBridge.gen()` → `scipy.optimize.minimize`

This document give an overview of the key concepts and flow when calling `ax.ModelBridge.gen()`.

Related resources:
- `docs\source\technical_details\ax_botorch\options_routing_guide.md`: Deepdive on passing optimisation arguments.
- `tutorials\ax_botorch\botorch_minimal_example_custom_acq.ipynb`: Minimal code example of custom acquisition function integration.

**Ax version**: 0.3.7
**BoTorch version**: as pinned by ax 0.3.7

NOTE: largely generate with AI, mid-level human review.

## Table of Contents
1. [Overview](#overview)
2. [Architecture Layers](#architecture-layers)
3. [Detailed Flow](#detailed-flow)
4. [Key Classes and Responsibilities](#key-classes-and-responsibilities)
5. [Optimization Parameter Pipeline](#optimization-parameter-pipeline)
6. [Code Examples](#code-examples)
7. [References](#references)

---

## Overview
The Ax/Botorch/Scipy layers have the following high level concerns when `.gen()` is called:
- **Ax**: Formulate the different objects Ax supports (e.g. data formats, search space types, transforms multi-objective, multi-fidelity, constraints, etc.) into a botorch acqusition function, which is then optimised.
- **BoTorch**: `AcquisitionFunction` definition, and associated higher level optimisation routines (Multi restart, initial conditions etc.)
- **Scipy**: Optimisation engine, performs the optimisation from a given point.

See `tutorials/ax_botorch/botorch_minimal_example_custom_acq.ipynb` for minimal code example of each of these steps.

### Conceptual Architecture
**Ax level** — transforms Ax objects into tensors, then creates and runs an `Acquisition` (Ax's internal wrapper around BoTorch components):

```
ModelBridge.gen(n, model_gen_options)        [ax.modelbridge.torch.TorchModelBridge]
    ↓
    BoTorchModel.gen()                       [ax.models.torch.botorch_modular.model.BoTorchModel]
        ├── Acquisition.__init__()           [ax.models.torch.botorch_modular.acquisition.Acquisition]
        │       └── botorch_acqf_class(...)  [botorch.acquisition.AcquisitionFunction subclass]
        └── Acquisition.optimize()
                └── optimize_acqf(...)       [botorch.optim.optimize]
```

- `Acquisition.__init__()`: exists to handle all Ax-specific logic around acquisition function construction (pending observations, multi-objective, constraints, multi-fidelity, etc.) before simply instantiating a `botorch.acquisition.AcquisitionFunction`.
    - BoTorch integration: Acquisition function must **register an input constructor** (`@acqf_input_constructor`). This selects the the relevant kwargs to instantiate the botrch acquistion function from a large set Ax provided.

- `Acquisition.optimize()` exists to handle Ax logic for different search space types (continuous, discrete, mixed) before simply calling `botorch.optim.optimize.optimize_acqf` (or its discrete equivalent).
    - BoTorch integration: Acquisition function must **register optimizer defaults** (`@optimizer_argparse.register`). This provides sensible default optimization parameters (e.g. `num_restarts`, `method`, `with_grad`) that can be overridden by the user.

**BoTorch level**:
- `botorch.acquisition.AcquisitionFunction`: has a `forward()` method returning the acquisition score for a candidate point
- `botorch.optim.optimize.optimize_acqf`: helper to optimise an `AcquisitionFunction` with multi-start and initial condition generation


### Code flow
```
User Code
    ↓
ModelBridge.gen(n, model_gen_options)
    ↓
TorchModelBridge._gen()
    ├── Transforms search space → SearchSpaceDigest
    ├── Builds TorchOptConfig (embeds model_gen_options)
    └── calls → BoTorchModel.gen()
                    ├── construct_acquisition_and_optimizer_options()
                    │       ├── acq_options  ← model_gen_options["acquisition_function_kwargs"]
                    │       └── opt_options  ← model_gen_options["optimizer_kwargs"]
                    ├── _instantiate_acquisition(acq_options)
                    │       └── Acquisition.__init__()
                    │               ├── get_acqf_input_constructor() → input_constructor
                    │               ├── input_constructor(**acq_options) → acqf_inputs
                    │               └── botorch_acqf_class(**acqf_inputs) → self.acqf
                    └── Acquisition.optimize(opt_options)
                            ├── optimizer_argparse(acqf, optimizer_options=opt_options)
                            └── optimize_acqf(**optimizer_options_with_defaults)
                                    └── _optimize_acqf_batch()
                                            ├── gen_batch_initial_conditions()
                                            └── gen_candidates_scipy()
                                                    └── minimize_with_timeout()
                                                            └── scipy.optimize.minimize()
```
---
## Key workflow to integrate botorch acquisition function into Ax Generation:
To use a custom BoTorch acquisition function in Ax, the following integration points are relevant:
- **Acquisition Function Construction (Mandatory)**: Register an input constructor that extracts the relevant kwargs for acqusition function instantiation from a large set passed by Ax.
- **Optimizer Defaults (recommended)**: Register default optimization parameters for your acquisition function, which can be overridden by user-provided options.

### Acquisition Function Construction
See `tutorials\ax_botorch\botorch_minimal_example_custom_acq.ipynb`, `construct_inputs_custom_acq` function for a minimal code example.

### Optimizer Defaults
See `tutorials\ax_botorch\botorch_minimal_example_custom_acq.ipynb`, `_argparse_custom_acquisition` function for a minimal code example.

---
## Details:
### Architecture Layers

#### 1. Entry Point: `ModelBridge.gen()`

**Location**: `ax.modelbridge.base.ModelBridge`

**Purpose**: User-facing entry point for generating new candidate points. Handles Ax-level transforms and validation before delegating to the model.

**Key Parameters**:
- `n`: Number of candidates to generate
- `search_space`: Search space (defaults to model's search space)
- `optimization_config`: Objective and constraints
- `pending_observations`: Points currently being evaluated
- `fixed_features`: Features to fix during generation
- `model_gen_options`: Dictionary controlling acquisition and optimization behaviour

**Key Behaviour**:
- Validates inputs
- Applies Ax transforms to search space and observations
- Calls `self._gen(...)` passing `model_gen_options` through unchanged
- Untransforms generated candidates back to problem space

#### 2. Bridge Layer: `TorchModelBridge._gen()`

**Location**: `ax.modelbridge.torch.TorchModelBridge`

**Purpose**: Converts Ax-level objects into tensor-based representations suitable for BoTorch, and packages everything into a `TorchOptConfig`.

**Key Behaviour**:
- Merges `self._default_model_gen_options` with user-provided `model_gen_options`
- Builds `SearchSpaceDigest` (bounds, feature types, etc.)
- Builds `TorchOptConfig` dataclass containing:
  - `objective_weights`, `outcome_constraints`, `linear_constraints`
  - `fixed_features`, `pending_observations`
  - `model_gen_options` (passed through as-is)
  - `rounding_func`
- Calls `self.model.gen(n, search_space_digest, torch_opt_config)`

#### 3. Model Layer: `BoTorchModel.gen()`

**Location**: `ax.models.torch.botorch_modular.model.BoTorchModel`

**Purpose**: Orchestrates the two main steps: acquisition function instantiation and optimization. Splits the user's options into the two relevant streams.

**Key Behaviour**:
- Calls `construct_acquisition_and_optimizer_options()` to split options
- Calls `_instantiate_acquisition()` with `acq_options`
- Calls `Acquisition.optimize()` with `opt_options`

#### 4. Acquisition Wrapper: `Acquisition`

**Location**: `ax.models.torch.botorch_modular.acquisition.Acquisition`

**Purpose**: Ax's wrapper around a BoTorch `AcquisitionFunction`. Handles instantiation (via input constructors) and optimization (via `optimize_acqf`).

**Key Attributes**:
- `acqf`: The actual BoTorch `AcquisitionFunction` instance
- `surrogates`: Dict of Surrogate model(s)
- `options`: The `acq_options` dict passed during construction
- `X_pending`: Pending points tensor
- `X_observed`: Observed points tensor

**Key Methods**:
- `__init__()`: Instantiates the BoTorch acquisition function via input constructor
- `optimize()`: Runs multi-start optimization of the acquisition function
- `evaluate()`: Evaluates the acquisition function at given points

#### 5. Optimizer Argument Parser: `optimizer_argparse`

**Location**: `ax.models.torch.botorch_modular.optimizer_argparse`

**Purpose**: A `Dispatcher` that resolves default optimization parameters based on the acquisition function type. Custom acquisition functions can register their own defaults.

**Key Behaviour**:
- Dispatches based on the acquisition function class (uses class hierarchy)
- Returns a dict of kwargs for `optimize_acqf`
- User-provided `optimizer_options` override the defaults

#### 6. BoTorch Optimization: `optimize_acqf()`

**Location**: `botorch.optim.optimize`

**Purpose**: Multi-start optimization of the acquisition function. Generates initial conditions, then runs local optimization from each start.

**Key Parameters**:
- `acq_function`: The BoTorch acquisition function
- `bounds`: Parameter bounds tensor `(2, d)`
- `q`: Number of candidates per batch
- `num_restarts`: Number of random restarts for multi-start optimization
- `raw_samples`: Number of initial quasi-random samples for generating starting points
- `options`: Dict passed to `gen_candidates_scipy` (and eventually to scipy)
- `sequential`: Whether to optimize candidates one at a time
- `gen_candidates`: Callable for local optimization (default: `gen_candidates_scipy`)

#### 7. Scipy Interface: `gen_candidates_scipy()`

**Location**: `botorch.generation.gen`

**Purpose**: Wraps the acquisition function for scipy consumption. Converts between torch tensors and numpy arrays, handles gradient computation.

**Key Parameters (via `options` dict)**:
- `method`: Scipy optimizer method (default: `"L-BFGS-B"` or `"SLSQP"` with constraints)
- `with_grad`: Whether to compute gradients (default: `True`)
- `maxiter`: Maximum iterations (default: `2000`)
- `callback`: Optional callback function
- Any other key is passed as scipy `options`

#### 8. Scipy: `minimize_with_timeout()`

**Location**: `botorch.optim.utils.timeout`

**Purpose**: Thin wrapper around `scipy.optimize.minimize` that adds timeout support via an injected callback.

---

### Flow

#### Step 1: User calls `ModelBridge.gen()`

```python
model_bridge.gen(n=1, model_gen_options={...})
```

#### Step 2: `TorchModelBridge._gen()` merges options and builds config

```python
augmented_model_gen_options = {
    **self._default_model_gen_options,
    **(model_gen_options or {}),
}
# Builds TorchOptConfig with model_gen_options embedded
search_space_digest, torch_opt_config = self._get_transformed_model_gen_args(...)
# Delegates to the model
gen_results = self.model.gen(n, search_space_digest, torch_opt_config)
```

#### Step 3: `BoTorchModel.gen()` splits options

```python
acq_options, opt_options = construct_acquisition_and_optimizer_options(
    acqf_options=self.acquisition_options,
    model_gen_options=torch_opt_config.model_gen_options,
)
```

**`construct_acquisition_and_optimizer_options`** (in `ax/models/torch/botorch_modular/utils.py`):
- `acq_options` = `self.acquisition_options` merged with `model_gen_options["acquisition_function_kwargs"]`
- `opt_options` = copy of `model_gen_options["optimizer_kwargs"]`

Key constants:
- `Keys.ACQF_KWARGS = "acquisition_function_kwargs"`
- `Keys.OPTIMIZER_KWARGS = "optimizer_kwargs"`

#### Step 4: `_instantiate_acquisition()` creates the Acquisition wrapper

```python
acqf = self._instantiate_acquisition(
    search_space_digest=search_space_digest,
    torch_opt_config=torch_opt_config,
    acq_options=acq_options,
)
```

This creates:
```python
return self.acquisition_class(
    surrogates=self.surrogates,
    botorch_acqf_class=self.botorch_acqf_class,  # e.g. QoILookAhead
    search_space_digest=...,
    torch_opt_config=...,
    options=acq_options,
)
```

#### Step 5: `Acquisition.__init__()` instantiates the BoTorch acquisition function

Three key sub-steps:

**5a. Look up the input constructor from BoTorch's registry:**
```python
input_constructor = get_acqf_input_constructor(botorch_acqf_class)
```
This finds the function registered with `@acqf_input_constructor(MyAcqf)` in `ACQF_INPUT_CONSTRUCTOR_REGISTRY`.

**5b. Call the input constructor with combined kwargs:**
```python
input_constructor_kwargs = {
    "X_baseline": unique_Xs_observed,
    "X_pending": unique_Xs_pending,
    "objective_thresholds": objective_thresholds,
    "constraints": ...,
    "target_fidelities": ...,
    "bounds": ...,
    **acqf_model_kwarg,   # {"model": model}
    **model_deps,
    **self.options,        # ← acq_options passed through here
}

acqf_inputs = input_constructor(
    training_data=training_data,
    objective=objective,
    posterior_transform=posterior_transform,
    **input_constructor_kwargs,
)
```

**5c. Instantiate the BoTorch acquisition function:**
```python
self.acqf = botorch_acqf_class(**acqf_inputs)
```

**Example — QoILookAhead:**

The registered input constructor is `construct_inputs_qoi_look_ahead` in `axtreme/acquisition/qoi_look_ahead.py`:
```python
@acqf_input_constructor(QoILookAhead)
def construct_inputs_qoi_look_ahead(model, qoi_estimator, sampler, **_):
    return {"model": model, "qoi_estimator": qoi_estimator, "sampler": sampler}
```
The `qoi_estimator` and `sampler` arrive via `self.options` → `acq_options` → `model_gen_options["acquisition_function_kwargs"]`.

#### Step 6: `Acquisition.optimize()` resolves optimizer defaults and calls `optimize_acqf`

```python
optimizer_options_with_defaults = optimizer_argparse(
    self.acqf,           # dispatches based on acqf type
    bounds=bounds,
    q=n,
    optimizer_options=optimizer_options,  # ← opt_options from step 3
)
```

**`optimizer_argparse`** is a `Dispatcher`. It dispatches based on the acquisition function class:

**Default (`_argparse_base`, registered for `AcquisitionFunction`):**
```python
@optimizer_argparse.register(AcquisitionFunction)
def _argparse_base(acqf, sequential=True, num_restarts=20, raw_samples=1024,
                   init_batch_limit=32, batch_limit=5, optimizer_options=None, ...):
    optimizer_options = optimizer_options or {}
    return {
        "sequential": sequential,
        "num_restarts": num_restarts,
        "raw_samples": raw_samples,
        "options": {
            "init_batch_limit": init_batch_limit,
            "batch_limit": batch_limit,
            **optimizer_options.get("options", {}),
        },
        **{k: v for k, v in optimizer_options.items() if k != "options"},
    }
```

**Custom (`_argparse_qoi_look_ahead` in axtreme):**
```python
@optimizer_argparse.register(QoILookAhead)
def _argparse_qoi_look_ahead(acqf, **kwargs):
    args = _argparse_base(acqf, **kwargs)
    # Override defaults: raw_samples=100, with_grad=False, method="Nelder-Mead"
    ...
    return args
```

Then calls:
```python
optimize_acqf(
    acq_function=self.acqf,
    bounds=bounds,
    q=n,
    inequality_constraints=...,
    fixed_features=...,
    post_processing_func=...,
    **optimizer_options_with_defaults,
)
```

#### Step 7: `optimize_acqf()` sets up multi-start optimization

- Sets `gen_candidates = gen_candidates_scipy` (default)
- Wraps all inputs into an `OptimizeAcqfInputs` dataclass
- Calls `_optimize_acqf()` → `_optimize_acqf_batch()`

#### Step 8: `_optimize_acqf_batch()` runs batched multi-start optimization

- Generates initial conditions via `gen_batch_initial_conditions()` (uses `raw_samples` and `num_restarts`)
- Splits initial conditions into batches of size `batch_limit`
- For each batch, calls:
  ```python
  gen_candidates(batched_ics_, acq_function,
      lower_bounds=..., upper_bounds=...,
      options={k: v for k, v in options.items() if k not in INIT_OPTION_KEYS},
      fixed_features=..., timeout_sec=...)
  ```

`INIT_OPTION_KEYS` are filtered out before passing to scipy:
```python
INIT_OPTION_KEYS = {
    "alpha", "batch_limit", "eta", "init_batch_limit", "nonnegative",
    "n_burnin", "sample_around_best", "sample_around_best_sigma",
    "sample_around_best_prob_perturb", "seed",
}
```

#### Step 9: `gen_candidates_scipy()` interfaces with scipy

- Reads from `options`:
  - `method`: default `"L-BFGS-B"` (no constraints) or `"SLSQP"` (with constraints)
  - `with_grad`: default `True`
  - `maxiter`: default `2000`
  - `callback`: optional
- If `with_grad=True`: wraps acqf in a numpy function that computes value + gradient via `torch.autograd.grad`
- If `with_grad=False`: wraps acqf computing value only (scipy uses finite differences)
- Calls:
```python
minimize_with_timeout(
    fun=f_np_wrapper,
    args=(f,),            # f = lambda x: -acquisition_function(x)
    x0=x0,
    method=options.get("method", "SLSQP" if constraints else "L-BFGS-B"),
    jac=with_grad,
    bounds=bounds,
    constraints=constraints,
    callback=options.get("callback", None),
    options={k: v for k, v in options.items()
             if k not in ["method", "callback", "with_grad"]},  # e.g. maxiter, maxfev
    timeout_sec=timeout_sec,
)
```

#### Step 10: `minimize_with_timeout()` calls scipy

```python
scipy.optimize.minimize(fun, x0, args, method, jac, bounds, constraints, callback, options)
```
with an injected timeout callback.

---

### Code Examples

#### Example 1: Basic Candidate Generation

```python
from ax.modelbridge.registry import Models

# Create model (see gp_model_creation_flow.md)
model_bridge = Models.BOTORCH_MODULAR(
    experiment=experiment,
    data=experiment.fetch_data(),
)

# Generate a candidate with default settings
generator_run = model_bridge.gen(n=1)
```

#### Example 2: Custom Acquisition Function with QoILookAhead

```python
from ax.modelbridge.registry import Models
from axtreme.acquisition.qoi_look_ahead import QoILookAhead
from axtreme.sampling import MeanSampler

model_bridge = Models.BOTORCH_MODULAR(
    experiment=experiment,
    data=experiment.fetch_data(),
    botorch_acqf_class=QoILookAhead,
)

# Generate with custom acquisition and optimizer options
generator_run = model_bridge.gen(
    n=1,
    model_gen_options={
        "acquisition_function_kwargs": {
            "qoi_estimator": my_qoi_estimator,
            "sampler": MeanSampler(),
        },
        "optimizer_kwargs": {
            "num_restarts": 10,
            "raw_samples": 100,
            "options": {
                "with_grad": False,
                "method": "Nelder-Mead",
                "maxfev": 50,
            },
        },
    },
)
```

#### Example 3: Registering a Custom Input Constructor

```python
from botorch.acquisition.input_constructors import acqf_input_constructor

@acqf_input_constructor(MyCustomAcqf)
def construct_inputs_my_acqf(model, my_param, **_):
    """Registered input constructor for MyCustomAcqf.

    Args are received from Acquisition.__init__ kwargs:
    - model: always provided by Ax
    - my_param: provided via model_gen_options["acquisition_function_kwargs"]["my_param"]
    """
    return {"model": model, "my_param": my_param}
```

#### Example 4: Registering Custom Optimizer Defaults

```python
from ax.models.torch.botorch_modular.optimizer_argparse import (
    optimizer_argparse,
    _argparse_base,
)

@optimizer_argparse.register(MyCustomAcqf)
def _argparse_my_acqf(acqf, **kwargs):
    """Custom optimizer defaults for MyCustomAcqf."""
    args = _argparse_base(acqf, **kwargs)

    optimizer_options = kwargs.get("optimizer_options", {})
    options = optimizer_options.get("options", {})

    # Set defaults (only if user hasn't provided them)
    if "raw_samples" not in optimizer_options:
        args["raw_samples"] = 50
    if "with_grad" not in options:
        args["options"]["with_grad"] = False
    if "method" not in options:
        args["options"]["method"] = "Nelder-Mead"

    return args
```

#### Example 5: Accessing the Acquisition Function Directly

```python
# After model creation
model_bridge = Models.BOTORCH_MODULAR(experiment=experiment, data=data)

# Access the underlying BoTorch model
botorch_model = model_bridge.model

# Manually instantiate acquisition for inspection
from ax.models.torch.botorch_modular.acquisition import Acquisition
acqf_wrapper = Acquisition(
    surrogates={"": botorch_model.surrogate},
    botorch_acqf_class=QoILookAhead,
    search_space_digest=search_space_digest,
    torch_opt_config=torch_opt_config,
    options={"qoi_estimator": my_qoi, "sampler": MeanSampler()},
)

# The actual BoTorch acquisition function
print(type(acqf_wrapper.acqf))  # → QoILookAhead
```

---

## References

- Ax Documentation: https://ax.dev/
- BoTorch Documentation: https://botorch.org/
- Code locations:
  - `ax.modelbridge.base.ModelBridge.gen()`: Entry point
  - `ax.modelbridge.torch.TorchModelBridge._gen()`: Bridge-level generation
  - `ax.models.torch.botorch_modular.model.BoTorchModel.gen()`: Model-level orchestration
  - `ax.models.torch.botorch_modular.utils.construct_acquisition_and_optimizer_options()`: Option splitting
  - `ax.models.torch.botorch_modular.acquisition.Acquisition`: Instantiation and optimization wrapper
  - `ax.models.torch.botorch_modular.optimizer_argparse`: Default optimizer arguments
  - `botorch.optim.optimize.optimize_acqf()`: Multi-start optimization
  - `botorch.generation.gen.gen_candidates_scipy()`: Scipy interface
  - `botorch.optim.utils.timeout.minimize_with_timeout()`: Scipy wrapper
  - `axtreme.acquisition.qoi_look_ahead`: Custom acquisition function example
