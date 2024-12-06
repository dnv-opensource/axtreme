# Example Usecase
The following offshore wind usecase is used to introduce the `RAX` package, as we have found it more informative than abstract technical definitions of key package terminology and capabilities:

## Overview
An offshore wind turbine is being designed to operate for 20 years. The engineers have information about the typical weather conditions (called the environment conditions), and have a slow but accurate stochastic simulator of the response experienced by the turbine in a give weather condition. In order to know their design will survive for the 20 year period they need to know (amongst other things) the Ultimate Limit State (ULS). This is the largest response the wind turbine is experiences in 20 years of operation.

Because of randomness in the weather and the turbine's response, the largest response experienced in a 20 year period is not a fixed values, and has a distribution. This distribution is called the  the Extreme Response Distribution (ERD). The engineers are specifically interested in knowing the median of this distribution (this is called their Quantity of Interest (QoI)).

![long_term_response_dist](img/usecase_offshorewind/axtreme_long_term_response_distribution.png)

### Environment Samples:
The environment represents the factors that effect the wind turbine, and these are used as inputs to the simulator. Typically these are weather conditions such as wind and wave information. The environment distribution then represents how likely it is that a condition will be experience by the wind turbine.

The environment distribution reports "long-term" conditions. This means it represents the average conditions (e.g wind speed) over an hour. This is different to the instantaneous conditions, which are called "short-term" conditions.

### Wind Turbine Simulator $f:X -> Y$
This stochastic function predicts the response caused by a given environment condition.
- Input: Takes a "long-term" environment condition $x$ . e.g average wind speed and turbulence.
- Output: The turbines maximum response $y$ during an hour of operation in those conditions.  The response is stochastic as the model internally randomly generates an hour on instantaneous weather that satisfies the long term average $x$. As there are many patterns of instantaneous weather, each which may produce different responses, the output is stochastic

## Possible Solutions:

### Naive approach:
If the simulator was fast we could use the following process to calculate the QoI:
- Taking one period worth of environment samples.
- Running them through the simulator, creating one period worth of responses.
- Taking the largest responses in that period.

This produces a single sample of the ERD. The process could be be repeated until enough samples were obtains to estimate the QoI.

### Traditional Approaches:
Method such as Environment Contouring have been create to approximate the QoI with fewer runs of the simulator. These work better than the Naive approach, but have shortcomings.


### RAX approach:
The `RAX` package deals with this problem by using a fast surrogate model in place of the real simulator when performing the QoI calculations. The process consists of 3 key steps (detailed further in "Basic Concepts"):
1) #### Build an uncertainty-aware Surrogate Model.
Once a small dataset has been created with the simulator (input and output pairs), a surrogate model can be fit to this dataset. For the surrogate to be useful it should:
- Properly capture the (possibly non-Gaussian) randomness in the simulators response.
- Be uncertainty aware: Using a surrogate model introduces uncertainty regarding the quality of the fit. Without knowing this uncertainty, it is difficult to trust the results calculated using the surrogate.

2) #### Calculate the QoI:
The surrogate model can now be used in place of the simulator to calculate the QoI. It is important that the uncertainty the surrogate has regarding the true model (the simulator) is propagated through to the QoI.

3) #### Reduce the uncertainty in the QoI:
We can reduce uncertainty in the QoI by adding data to the surrogate (which reduces the surrogates uncertainty). Some data points are more influential in the QoI calculation than others. By using active learning (also called DoE) to identify these regions, uncertainty in the QoI can be reduced with minimal runs of the (expensive) simulator.

