# Compatibility with `Ax`
`Ax` is a package developed by Meta for orchestration of Bayesian Optimisation. By utilising `Ax`, a large amount of orchestration and book-keeping is automatically handled, minimising developer overhead and providing users with an established and documented interface. While there are a large selection of tools available for maximising a GP, our ULS (detailed in `usecase_offshorewind`) problem does not fit directly into the framework. We do the following to deal with this:
- Define the `Experiment` `Metrics` to be the parameters of the response distribution (e.g location and scale).
- When a `ModelBridge` is instantiated using an `ax` `Experiment` it automatically sets up GPs to track the `Metrics`.
- We customise the Acquisition function in the `ModelBridge` so that it appropriately handles our GPs (Typical acquisition functions try to find the maximum of the GP, but we are interested in a different Quantity)

The following shows a conceptual overview of the key `ax` components involved.

![axtreme_ax_component_diagram](img/ax_integration/ax_component_diagram.png)
