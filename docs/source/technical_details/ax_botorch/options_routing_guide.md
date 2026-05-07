# Options Routing Guide: `ModelBridge.gen()` → `scipy.optimize.minimize`

This document explains how to get your options to the right place when calling `ModelBridge.gen()`. The options dict is split and transformed multiple times between the Ax entry point and the underlying scipy optimizer.

Related resources:
- `model_bridge_gen_guide.md`: Overview of `ModelBridge.gen()`.
- `tutorials\ax_botorch\botorch_minimal_example_custom_acq.ipynb`: Creating a custom acquisition function in Botorch, optimising it, and intergrating it with Ax.Breakpoints can be set here to easily trace/play with the options flow.
- `tutorials\ax_botorch\optimisation.ipynb`: Deepdive on `optimize_acqf` settings.

NOTE: largely generate with AI, with mid-level human review.

## Table of Contents

1. [TL;DR — Where to put your options](#tldr--where-to-put-your-options)
2. [Detailed Routing](#detailed-routing)
   - [Level 1: ModelBridge.gen()](#level-1-modelbridgegenmodel_gen_options)
   - [Level 2a: Acquisition Function Construction](#level-2a-acquisition-function-construction)
   - [Level 2b: Optimization — optimizer_argparse](#level-2b-optimization--optimizer_argparse)
   - [Level 3: optimize_acqf → _optimize_acqf_batch](#level-3-optimize_acqf--_optimize_acqf_batch)
   - [Level 4: gen_candidates_scipy](#level-4-gen_candidates_scipyoptions)
   - [Level 5: scipy.optimize.minimize](#level-5-scipyoptimizeminimizeoptions)
3. [Complete Routing Diagram](#complete-routing-diagram)
4. [Practical Examples](#practical-examples)
5. [Key Gotchas](#key-gotchas)

## TL;DR — Where to put your options

```python
model_bridge.gen(
    n=1,
    model_gen_options={
        # ─── Goes to AcquisitionFunction.__init__() ───
        "acquisition_function_kwargs": {
            # Any kwarg your input_constructor expects
            "my_param": value,
        },
        # ─── Goes to optimize_acqf() (via optimizer_argparse) ───
        "optimizer_kwargs": {
            # Top-level kwargs of optimize_acqf:
            "num_restarts": 10,
            "raw_samples": 100,
            "sequential": True,
            "timeout_sec": 60.0,
            "retry_on_optimization_warning": False,

            # ─── Goes to gen_candidates_scipy() ───
            "options": {
                # Init-phase options (consumed by gen_batch_initial_conditions):
                "init_batch_limit": 32,
                "batch_limit": 5,
                "seed": 42,
                "sample_around_best": True,

                # gen_candidates_scipy options (consumed before scipy):
                "with_grad": False,
                "method": "Nelder-Mead",
                "callback": my_callback_fn,

                # ─── Goes to scipy.optimize.minimize(options={...}) ───
                "maxiter": 500,
                "maxfev": 1000,
                "xatol": 1e-8,
                "fatol": 1e-8,
                # ... any scipy method-specific option
            },
        },
    },
)
```

---

## Detailed Routing

### Level 1: `ModelBridge.gen(model_gen_options={})`

Code objects: `ax.modelbridge.base.ModelBridge` → `ax.modelbridge.torch.TorchModelBridge._gen()` → `ax.models.torch.botorch_modular.model.BoTorchModel.gen()`

The `model_gen_options` dict is passed unchanged through `TorchModelBridge._gen()` into `BoTorchModel.gen()`, where it is split by `construct_acquisition_and_optimizer_options()`:

```python
# ax.models.torch.botorch_modular.utils.construct_acquisition_and_optimizer_options
def construct_acquisition_and_optimizer_options(acqf_options, model_gen_options):
    acq_options = acqf_options.copy()                                    # from BoTorchModel.acquisition_options
    acq_options.update(model_gen_options["acquisition_function_kwargs"])  # merged in
    opt_options = model_gen_options["optimizer_kwargs"].copy()
    return acq_options, opt_options
```

| Key in `model_gen_options`        | Destination                          |
|----------------------------------|--------------------------------------|
| `"acquisition_function_kwargs"`  | → `Acquisition.__init__()` (acqf construction) |
| `"optimizer_kwargs"`             | → `Acquisition.optimize()` (optimization)      |

---

### Level 2a: Acquisition Function Construction

code object: `ax.models.torch.botorch_modular.acquisition.Acquisition.__init__()`

`acq_options` flows into `Acquisition.__init__()` where it is passed as `**self.options` to the registered **input constructor**:

```python
input_constructor_kwargs = {
    "model": model,
    "X_baseline": ...,
    "X_pending": ...,
    **self.options,        # ← your "acquisition_function_kwargs" end up here
}
acqf_inputs = input_constructor(**input_constructor_kwargs)
self.acqf = botorch_acqf_class(**acqf_inputs)
```

**Guidance**: Put anything your `@acqf_input_constructor` needs in `"acquisition_function_kwargs"`.

---

### Level 2b: Optimization — `optimizer_argparse`

Code object: `ax.models.torch.botorch_modular.acquisition.Acquisition.optimize()` → `ax.models.torch.botorch_modular.optimizer_argparse.optimizer_argparse`

`opt_options` is passed as `optimizer_options` to `optimizer_argparse()`, which produces the final kwargs for `optimize_acqf`:

```python
# ax.models.torch.botorch_modular.acquisition.Acquisition.optimize()
optimizer_options_with_defaults = optimizer_argparse(
    self.acqf,
    bounds=bounds,
    q=n,
    optimizer_options=opt_options,  # ← your "optimizer_kwargs"
)
optimize_acqf(acq_function=self.acqf, bounds=bounds, q=n, **optimizer_options_with_defaults)
```

The default `_argparse_base` merges your options like this:

```python
# ax.models.torch.botorch_modular.optimizer_argparse._argparse_base
def _argparse_base(acqf, sequential=True, num_restarts=20, raw_samples=1024,
                   init_batch_limit=32, batch_limit=5, optimizer_options=None, **ignore):
    optimizer_options = optimizer_options or {}
    return {
        "sequential": sequential,
        "num_restarts": num_restarts,
        "raw_samples": raw_samples,
        "options": {
            "init_batch_limit": init_batch_limit,
            "batch_limit": batch_limit,
            **optimizer_options.get("options", {}),   # ← your "optimizer_kwargs"]["options"]
        },
        **{k: v for k, v in optimizer_options.items() if k != "options"},  # ← top-level overrides
    }
```

**Result**: Anything you put at the top level of `"optimizer_kwargs"` (except `"options"`) overrides top-level `optimize_acqf` kwargs. Anything inside `"optimizer_kwargs"["options"]` goes into the `options` dict passed down the stack.

---

### Level 3: `optimize_acqf()` → `_optimize_acqf_batch()`

Code object:`botorch.optim.optimize.optimize_acqf()` → `botorch.optim.optimize._optimize_acqf_batch()`

`optimize_acqf` accepts these relevant parameters (which come from `optimizer_options_with_defaults`).
#### Parameters
See docs [here](https://botorch.readthedocs.io/en/latest/optim.html#botorch.optim.optimize.optimize_acqf). Refer to related resource for a deepdive on `optimize_acqf` settings.:

| Parameter | Source | Purpose |
|-----------|--------|---------|
| `num_restarts` | top-level `"optimizer_kwargs"` | Number of multi-start restarts |
| `raw_samples` | top-level `"optimizer_kwargs"` | Quasi-random samples for initial conditions |
| `sequential` | top-level `"optimizer_kwargs"` | Sequential vs joint q-batch optimization |
| `timeout_sec` | top-level `"optimizer_kwargs"` | Total optimization timeout |
| `retry_on_optimization_warning` | top-level `"optimizer_kwargs"` | Retry on failure |
| `options` | `"optimizer_kwargs"["options"]` | Passed to init + gen_candidates |

#### Internal Functionality
Inside `_optimize_acqf_batch`, the `options` dict is split:

```python
# Some keys are used for gen_batch_initial_conditions, then FILTERED OUT:
INIT_OPTION_KEYS = {
    "alpha",
    "batch_limit",
    "eta",
    "init_batch_limit",
    "nonnegative",
    "n_burnin",
    "sample_around_best",
    "sample_around_best_sigma",
    "sample_around_best_prob_perturb",
    "seed",
    "thinning",
}

# What gets passed to gen_candidates_scipy:
gen_kwargs = {
    "options": {k: v for k, v in options.items() if k not in INIT_OPTION_KEYS},
    ...
}
```

---

### Level 4: `gen_candidates_scipy(options={})`

Code object: `botorch.generation.gen.gen_candidates_scipy()`

This function consumes three keys from its `options` dict and passes the rest to scipy:
#### Parameters consumed from 'options' argument:

| Key | Consumed by | Purpose |
|-----|-------------|---------|
| `"with_grad"` | `gen_candidates_scipy` | If `True`: computes gradients, sets `jac=True`. If `False`: no gradients, scipy uses finite differences |
| `"method"` | `gen_candidates_scipy` | Passed as `method` arg to `scipy.optimize.minimize`. Default: `"L-BFGS-B"` (no constraints) or `"SLSQP"` (with constraints) |
| `"callback"` | `gen_candidates_scipy` | Passed as `callback` arg to `scipy.optimize.minimize` |

Everything else becomes scipy's `options` dict:

#### Internal functionality

```python
# botorch.optim.utils.timeout.minimize_with_timeout → scipy.optimize.minimize
res = minimize_with_timeout(
    fun=f_np_wrapper,
    x0=x0,
    method=options.get("method", "SLSQP" if constraints else "L-BFGS-B"),
    jac=with_grad,
    bounds=bounds,
    constraints=constraints,
    callback=options.get("callback", None),
    options={k: v for k, v in options.items() if k not in ["method", "callback", "with_grad"]},
)
```

---

### Level 5: `scipy.optimize.minimize(options={})`

Code object: `scipy.optimize.minimize()`

What arrives here depends on the method. Common options per method:

| Method | Useful `options` keys |
|--------|----------------------|
| `L-BFGS-B` | `maxiter`, `maxfun`, `ftol`, `gtol`, `maxcor`, `maxls` |
| `Nelder-Mead` | `maxiter`, `maxfev`, `xatol`, `fatol`, `adaptive` |
| `SLSQP` | `maxiter`, `ftol`, `eps` |

See [scipy.optimize.minimize docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html) for method-specific options.

---

## Complete Routing Diagram

```
model_gen_options
├── "acquisition_function_kwargs"
│   └── → Acquisition.__init__() → input_constructor(**kwargs) → AcqFunction(**inputs)
│
└── "optimizer_kwargs"
    ├── "num_restarts"       → optimize_acqf(num_restarts=...)
    ├── "raw_samples"        → optimize_acqf(raw_samples=...)
    ├── "sequential"         → optimize_acqf(sequential=...)
    ├── "timeout_sec"        → optimize_acqf(timeout_sec=...)
    ├── "retry_on_..."       → optimize_acqf(retry_on_optimization_warning=...)
    │
    └── "options"
        ├── "init_batch_limit"   → gen_batch_initial_conditions (then filtered out)
        ├── "batch_limit"        → _optimize_acqf_batch (then filtered out)
        ├── "seed"               → gen_batch_initial_conditions (then filtered out)
        ├── "sample_around_best" → gen_batch_initial_conditions (then filtered out)
        │   ... (all INIT_OPTION_KEYS filtered before gen_candidates_scipy)
        │
        ├── "with_grad"    → gen_candidates_scipy (consumed, not passed to scipy)
        ├── "method"       → scipy.optimize.minimize(method=...)
        ├── "callback"     → scipy.optimize.minimize(callback=...)
        │
        └── <everything else> → scipy.optimize.minimize(options={...})
            ├── "maxiter"
            ├── "maxfev"
            ├── "ftol"
            ├── "xatol"
            └── ...
```

---

## Practical Examples

### Example 1: Gradient-free optimization with Nelder-Mead

Use this when your acquisition function doesn't support gradients (e.g. involves non-differentiable operations):

```python
model_gen_options={
    "optimizer_kwargs": {
        "num_restarts": 10,
        "raw_samples": 100,
        "options": {
            "with_grad": False,
            "method": "Nelder-Mead",
            "maxfev": 200,       # scipy option: max function evaluations
            "xatol": 1e-6,       # scipy option: absolute x tolerance
            "fatol": 1e-6,       # scipy option: absolute f tolerance
        },
    },
}
```

### Example 2: Faster optimization with fewer restarts

```python
model_gen_options={
    "optimizer_kwargs": {
        "num_restarts": 5,       # fewer multi-start points
        "raw_samples": 50,       # fewer initial samples
        "options": {
            "batch_limit": 5,    # how many restarts run in parallel
            "maxiter": 100,      # fewer scipy iterations per restart
        },
    },
}
```

### Example 3: L-BFGS-B with custom tolerances

```python
model_gen_options={
    "optimizer_kwargs": {
        "options": {
            "with_grad": True,     # default, uses autograd
            "method": "L-BFGS-B",  # default when no constraints
            "maxiter": 500,
            "ftol": 1e-9,          # scipy L-BFGS-B: function tolerance
            "gtol": 1e-6,          # scipy L-BFGS-B: gradient tolerance
        },
    },
}
```

### Example 4: Using `acquisition_options` on BoTorchModel (static options)

Options that don't change between `.gen()` calls can be set at model creation:

```python
model_bridge = Models.BOTORCH_MODULAR(
    experiment=exp,
    data=data,
    botorch_acqf_class=MyAcqf,
    acquisition_options={"my_static_param": value},  # merged into acq_options
)
# At gen time, "acquisition_function_kwargs" is merged ON TOP of acquisition_options
model_bridge.gen(n=1, model_gen_options={
    "acquisition_function_kwargs": {"my_dynamic_param": other_value},
})
```

### Example 5: Custom optimizer defaults via `optimizer_argparse`

Register defaults so users don't need to specify them every `.gen()` call:

```python
from ax.models.torch.botorch_modular.optimizer_argparse import optimizer_argparse, _argparse_base

@optimizer_argparse.register(MyCustomAcqf)
def _argparse_my_acqf(acqf, **kwargs):
    args = _argparse_base(acqf, **kwargs)

    # Override defaults (respect user overrides via optimizer_options)
    user_options = kwargs.get("optimizer_options", {})
    user_inner_options = user_options.get("options", {})

    if "raw_samples" not in user_options:
        args["raw_samples"] = 50
    if "with_grad" not in user_inner_options:
        args["options"]["with_grad"] = False
    if "method" not in user_inner_options:
        args["options"]["method"] = "Nelder-Mead"

    return args
```

---

## Key Gotchas

1. **`"options"` nesting**: Options inside `"optimizer_kwargs"["options"]` serve *three different consumers* (init, gen_candidates_scipy, scipy). The routing is determined by key name, not explicit targeting.

2. **`optimizer_argparse` sets defaults**: If you don't provide `num_restarts` etc., they come from `_argparse_base` (or your registered override), not from `optimize_acqf` defaults. The argparse output *replaces* `optimize_acqf`'s defaults because it's splatted as `**kwargs`.

3. **`acquisition_options` vs `"acquisition_function_kwargs"`**: Both end up in the same place. `acquisition_options` is set at model construction time; `"acquisition_function_kwargs"` is per-`.gen()` call and takes precedence (dict `.update()`).

4. **INIT_OPTION_KEYS are silently consumed**: If you put `"batch_limit"` in options, it controls batch size during optimization but is NOT passed to scipy. This is by design but can be confusing.

5. **Float64 matters**: When using gradient-based optimization with scipy, use `torch.float64`. Float32 causes scipy to recommend tiny steps that get truncated, leading to no progress.

6. **`"method"` determines valid scipy options**: Each scipy method accepts different `options` keys. Passing an unsupported key is silently ignored by some methods and raises errors in others. Check [scipy docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html) for your chosen method.
