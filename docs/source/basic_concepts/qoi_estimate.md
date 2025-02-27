# QoI Estimates
We create a surrogate model in order to better understand the Extreme Response Distribution (ERD). Typically we are interested in a specific quantity of this distribution (e.g the median, or the 90th percentile), which we call the Quantity of Interest (QoI).

It is important that we can propagate the uncertainty in our surrogate model through to our QoI. The following figure introduced in "Surrogate Model" can help us gain an intuitive understanding of how this is done.

![axtreme_surrogate_model_uncertainty_aware](img/surrogate_model/surrogate_model_uncertainty_aware.png)

This diagram shows an uncertainty aware surrogate model. The grey lines represent different functions the surrogate believes could be the true simulator function (e.g The functions that could have generated the data it has seen). The diversity of the grey lines represent the uncertainty that the surrogate model has. A simple way to propagate this uncertainty through to the QoI is to perform the QoI calculation for each of the grey lines. This will give a variety of QoI values, and this distribution captures how the uncertainty in the surrogate propagates through to the QoI.

The is represented in `axtreme` by the `QoIEstimator` protocol. This takes a surrogate model, from which it samples possible functions (the grey lines), and returns the QoI estimated by each of the possible functions sampled.
