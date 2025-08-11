# Introduction
## What is axtreme (Ax for Extremes)
`axtreme` is a toolkit for estimating the long term behaviour (extremes) of expensive, stochastic, black box functions. It does this by creating uncertainty aware surrogate models of the black box function, propagating this uncertainty through to the behaviour being estimated (e.g produce estimates with confidence bounds), and using Active Learning (also know as Design of Experiments (DOE)) to iteratively improve the surrogate model and associated estimates.

This following presentation gives an overview of the key concepts and benefits of `axtreme`. It was part of the Industrial AI for Safety-Critical Systems Seminar in June 2025 hosted by DNV.

<iframe width="100%" height="600"
        src="https://players.brightcove.net/5836955873001/U0es4DTHP_default/index.html?videoId=6374738718112&autoplay=false"
        title="Bayesian Surrogate Modelling and Optimization of Extreme Response Calculation"
        frameborder="0"
        allow="picture-in-picture"
        allowfullscreen>
</iframe>

## What problems is it for
While there are a broad range of applications, development has been motivated by challenges commonly found in Engineering. Specifically, Ultimate Limit State (ULS) calculations often require engineers to understand a structure/design's performance over many years of operation. The models engineers use (FEM etc.) are typically very accurate, but slow, making it challenging to use them over this time frame. An illustrative use case is provided in "Example usecase".

## Why use it
Within engineering, probabilistic approaches to these types of problems can offer advantages over traditional methods (for example, demonstrated [here](https://www.sciencedirect.com/science/article/abs/pii/S0951833920300745)). In particular, they can help reduce conservatism in designs. Despite the advantages, probabilistic methods are not widely used, likely due in part to the complexity of implementing them (often requiring specialist knowledge). `axtreme` is designed to reduce this complexity and make probabilistic approaches more accessible, unlocking better understanding of long term behaviour, and potential reductions in excessive conservatism.

axtreme itself is built on [Ax](https://ax.dev/docs/why-ax.html) and [Botorch](https://botorch.org/docs/introduction). This provides out of the box access to best practice Gaussian Processes (GP), experiment management, auto-differentiation and GPU acceleration.

## Target Audience
`axtreme` is intended for users and engineers that are familiar with Python and Tensors. Familiarity with Bayesian Optimisation, Gaussian Processes and Acquisition functions is not assumed.
