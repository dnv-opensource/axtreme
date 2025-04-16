# Glossary
## Key terms
- **Extreme Response Distribtuion (ERD):** Distribution of the largest response experienced over a timeframe. e.g distribution of the largest response a windturbine will experience in 20 years of operation.
- **period**: A sample of the environment for the timeframe of interest. e.g 20 years worth of samples of the env dist.
    - `n_periods` The number of these periods.
    - `period_len`: the number of samples required to create the timeframe of interest.
        - e.g timeframe is 1 year, and env samples are for 1 day, then `period_len=365.25`
- **environment distribution (env_dist)**:
- **problem space**: Refers to the input variables and responses before any transforms/standardization have occurred. For example `ax.ModelBridge` operates in the problem space because it takes raw/untransformed x value and produce raw/untransformed y values.
- **model space**: Refers to the input variables and responses **after** transforms/standardization have occurred. Typically $x$ inputs are scaled to unit hypercube, and y values are standardised. For example `ax.Model` (found at `ModelBridge.model.surrogate.model`) operate in this space.

## Dimension Notation:
The `ax` stack (`ax`,  `botorch`, `gpytorch`, `pytorch`) comprises of a number of libraries, each with their own notation. As `axtreme` interacts with differrent parts of this stack, it is useful to know the different conventions. `axtreme` uses `botorch` tensor notation unless otherwise specified.

### Botorch tensor notation.
key terms that we make use of that we should define:
- Dimension convension used by (botorch)[https://botorch.org/api/models.html#botorch.models.gp_regression.SingleTaskGP]
    - `X`:  input data
    - `batch_shape`: (*b) batch shape. Varying number of dimensions (including 0)
    - `n`: input points.
    - `m`: target/output dimensionality
    - `d`: dimensionality of input points.
- Optimisation:
    - `q`: number of candidate points optimised jointly.
    - `t`: number of points passed to optimise in parrallel (not optimised jointly)

###
- The dimension convension used by gpytorch
    - (`...`,` b1 x ... x bk`): batch shape
    - `n`: input points.
    - `t`: target/output dimensionality
    - `d`: dimensionality of input points.

## TODO
Glossary task to be added

Explain how:
-  Multitask in gpytorch can be used with SingleTaskGP