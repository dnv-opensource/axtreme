# QoILookAhead design decisions

## Types of models handles

In the `QoILookAhead` acquisition function we need to get the training inputs and training target observation noise from the model. However, this is in different formats for different kinds of models. This makes the fetching and standardizing of this data challenging.

We therefore decided to figure out what kind of model we would commonly need to handle in the acquisition function by seeing what types of models that can result from an ax experiment. When experimenting with different cases of configuration of the ax experiment and using `ax.modelbridge.registry.Models.BOTORCH_MODULAR` as shown below.

```python
ax_model = Models.BOTORCH_MODULAR(
    search_space=experiment.search_space,
    experiment=experiment,
    data=experiment.fetch_data())

botorch_model = ax_model.model.surrogate.model
```

However, when doing this the resulting `botorch_model` was always of the type `SingleTaskGP`. Since that was the only type that resulted from this we decided to only handle this type of model in `QoILookAhead`.

If this assumption is wrong the `__init__` function will end up throwing an error. Then `QoILookAhead` should be modified to handle this unexpected case.

### Training inputs

With this assumption `model.train_inputs` was always a tuple containing a single `torch.Tensor`.
This `torch.Tensor` was of the shape `(num_targets, num_points, num_features)`, unless `num_targets = 1`. In that case the shape would be `(num_points, num_features)`.

### Observation noise

With this assumption the observation noise can always be found by calling `model.likelihood.noise` which returns a `torch.Tensor`. This tensor can have different shape depending on the parameters:

 - `(num_points, )`
    - if `num_targets = 1`
 - `(num_targets, 1)`
    - if `num_points = 1`
    - or if there is an assumption that all observations have the same noise.
      - E.g. the target noise is set to `None` in some way. Then the observation noise is estimated and assumed equal for all points.
 - `(num_targets, num_points)`
    - Otherwise
