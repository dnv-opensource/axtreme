# Example for How to Set Up a Project for Axtreme

This example demonstrates how to use the `axtreme` module. The simulator used here is intentionally simplified — the goal is to showcase the **workflow and capabilities of `axtreme`**.

The Quantity of Interest (QoI) for this example is the $\exp(-1)$ quantile.

## Terminology Mapping: Axtreme vs. Use Case
| Axtreme Term       | Use Case Term                     |
|--------------------|-----------------------------------|
| Extreme Response   | e.g. 100-year maximum wave crest height |

More are given in [problem.py](problem.py).

## Key Components Overview & Interactions

| Component              | File                          | Description |
|------------------------|-------------------------------|-------------|
| **Main Orchestration** | `problem.py`                  | Central script tying all parts together:<br>1. Defines the surrogate model’s search space<br>2. Definition of the distribution that best captures the simulators response<br>3. Loads the simulator<br>4. Loads and processes environment data<br>5. Computes brute force solution<br>6. Computes importance samples and weights<br>7. Calculates the quantity of interest (QoI) |
| **Environment Data**   | `usecase/env_data.py`         | Defines all parameters for generating the environmental input data. Also provides functionality to generate and access this data. Visualization available in `usecase/eda.py`. |
| **Simulator**          | `simulator.py`                | Defines a simplified simulator for the given environment data. |
| **Brute Force Solver** | `brute_force.py`              | Computes a brute force reference solution which is used to benchmark the accuracy of the `axtreme`’s QoI estimate. This is not possible for most use cases.|
||`brute_force_loc_and_scale_estimates.py`|Helper functions for computing underlying distribution functions for the Gaussian process fit comparison|
| **Importance Sampling**| `create_importance_samples.py`| Computes importance samples and weights to efficiently estimate the QoI. Uses the environment distribution defined in `usecase/env_data.py`. A visual representation of the distribution is available in `usecase/data/hs_tp_pdf.png`. |
| | `importance_sampling.py`| Helper functions for computing the importance samples and weights.|
| **QoI Analysis**       | `qoi_bias_var.py`             | Provides tools to analyze bias and variance of different the QoI estimator for different hyperparameters. Used to select hyperparameters for the final QoI estimator in `problem.py`. |
| **Design of Experiments (DOE)** | `doe.py`            | Perform DoE, i.e. the process of adding optimal new env. data points to the surrogate model. |

## Additional Information
- For a detailed description of the different axtreme components check out tutorials/basic_example.py.

<!-- (TODO: (ak 21.11.25): Add reference to minimal example once it is added) -->
