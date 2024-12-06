"""QoILookAhead acquisition function that looks ahead at possible models and optimize according to a quantity of interest."""  # noqa: E501

import warnings
from contextlib import ExitStack
from typing import Any

import torch
from ax.models.torch.botorch_modular.optimizer_argparse import _argparse_base, optimizer_argparse
from botorch.acquisition import AcquisitionFunction
from botorch.acquisition.input_constructors import acqf_input_constructor
from botorch.models import SingleTaskGP
from botorch.models.model import Model
from botorch.utils.transforms import t_batch_mode_transform
from gpytorch.likelihoods.gaussian_likelihood import FixedNoiseGaussianLikelihood, GaussianLikelihood

from axtreme.qoi.qoi_estimator import QoIEstimator
from axtreme.sampling import MeanSampler, PosteriorSampler
from axtreme.utils.model_helpers import get_target_noise_singletaskgp, get_training_data_singletaskgp


class QoILookAhead(AcquisitionFunction):
    """QoILookAhead is a generic acquisition function that estimates the usefulness of a design point.

    It estimates the usefulness of a design point by:

    - Creating a new model(s) (e.g the looking ahead) by including the design point in the training data.

      .. Note::
        The new model condition on the design point, it does not update hyperparameters.

    - Calculate the QoI with the new model, and find the variance of this distribution.

    The design point that results in the lowest variance QoI is considered the most desireable.

    Note on Optimisation:
        Optimising the AcquisitionFunction is an important part of the DoE process. The stochasticity and smoothness of
        the acquisition function determine what optimisers can be used. This acquisition function has the following
        properties with the default setup:

    Smoothness:
        QoILookAhead is smooth (twice differentiable) if:

        - The model used for instantiation produces smooth outputs.
        - A sampler produces smooth y values.
        - The method used to produce y_var estimates is smooth.
        - The QoIEstimator is smooth with respect to the model (e.g small changes in the model produce smooth
          change in the QoIEstimator result.)

        The default optimiser assume the QoI may not be smooth, and uses a gradient free optimiser. If your QoI is
        smooth these setting should be overridden.

    Stochasticity:
        QoILookAhead is deterministic if all the above components are deterministic.
        With default settings it is deterministic.
    """

    def __init__(
        self,
        model: SingleTaskGP,
        qoi_estimator: QoIEstimator,
        sampler: PosteriorSampler | None = None,
    ) -> None:
        """QoILookAhead acquisition function.

        Args:
            model: A fitted model. Only SingleTaskGP models are currently supported.

                - General GpytorchModel may eventually be supported.

            qoi_estimator: A callable conforming to QoIEstimator protocol.
            sampler: A sampler that is used to sample fantasy observations for each candidate point.
                If None, a MeanSampler is used. This then uses the mean of the posterior as the fantasy observation.

                .. Note::
                    Sampler choice can effect the stochasticty and smoothness of the acquisition function. See class
                    docs for details.
        """
        super().__init__(model)
        self.qoi = qoi_estimator

        if sampler is None:
            sampler = MeanSampler()
        self.sampler = sampler

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: torch.Tensor) -> torch.Tensor:  # noqa: N803  # pyright: ignore[reportIncompatibleMethodOverride] (necessary due to `t_batch_mode_transform` decorator)
        """Forward method for the QoI acquisition function.

        For each candidate point in x this acquisition function does the following:

        - Fantasizes (via the chosen sampler) possible new observations (y) at this candidate point.
        - Trains a new GP with the additional (x,y_i) pair for each fantasy observation y_i.
        - Evaluates the qoi_estimator on these new GPs, and reports the variance in the resulting estimates.

        This optimization will pick the point that resulted in the lowest mean variance in the QoI estimates.

        Args:
            X: (t_batch, 1, d) input points to evaluate the acquisiton function at.

        Returns:
            The output of the QoI acquisition function that will be optimized with shape (num_points,).

        Todo:
            - This should be updated to use the `FantasizeMixin`. This is the botorch interface indicating
              `model.fantasize(X)` is supported, which does a large chunk of the functionality below. The challenge is
              `model.fantasize(X)` adds an additonal dimension at the start of all posteriors calculated e.g.
              `(num_fantasies, batch_shape, n, m)`. Unclear if our QoI methods can handle/respect the `num_fantasies`
              dimension, of if the different fantasy models can easily be extracted. Revisit at a future date.
            - Consider making the jobs batchable or multiprocessed.
        """
        # shape is: (t_batch, d)
        x = X.squeeze(1)
        x_grad = x.requires_grad

        if x_grad:
            msg = (
                "Gradient tracking is not yet supported, please set `no_grad=True` in optimizer settings."
                "See tutorials/ax_botorch/botrch_minimal_example_custom_acq.ipynb for details"
            )
            raise NotImplementedError(msg)

        # .model.posterior(X) turns gradient back on unless torch.no_grad is used.
        with ExitStack() as stack:
            # sourcery skip: remove-redundant-if
            if not x_grad:
                stack.enter_context(torch.no_grad())

            reject_if_batched_model(
                self.model
            )  # the posterior predictions used in forward don't support batched-models
            posterior = self.model.posterior(x)
            # shape is:  (n_posterior_samples, t_batch, m)
            y = self.sampler(posterior)

            # shape is: (t_batch, m)
            yvar = _get_fantasy_observation_noise(self.model, x)

            # extend the shapes so they all match (n_postior_samples, t_batch, m)
            x = x.expand(y.shape[0], -1, -1)
            yvar = yvar.expand(y.shape[0], -1, -1)

            # shape is: (n_posterior_samples, t_batch)
            lookahead_var = self._batch_lookahead(x, y, yvar)

            # the results of the different y samples need to be aggregated.
            # Use a bespoke aggregator if provided by the sampler, otherwise use the mean
            if hasattr(self.sampler, "mean"):  # noqa: SIM108
                lookahead_var = self.sampler.mean(dim=0)  # type: ignore  # noqa: PGH003
            else:
                lookahead_var = lookahead_var.mean(dim=0)

            # The output from an acquisition function is maximized, so we negate the QoI variances.
            lookahead_var = -lookahead_var

            if x_grad and not lookahead_var.requires_grad:
                msg = (
                    "Losing gradient in QoiLookAhead.forward()\n. Optimisation setting currently require gradient."
                    " This is likely due to gradient not propagated through the qoi_estimator."
                )
                raise RuntimeWarning(msg)

            return lookahead_var

    def _batch_lookahead(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        yvar: torch.Tensor,
    ) -> torch.Tensor:
        """Process a batch of lookahead points.

        Args:
            x (`*b`,n,d): x points to proceess
            y (`*b`,n,m): y points to process
            yvar (`*b`,n,m): yvar points to process

        Returns:
            torch.Tensor (`*b`,n) of lookahead results.
        """
        batch_shape = x.shape[:-2]
        n, d = x.shape[-2:]
        m = y.shape[-1]

        x_flat = x.reshape(-1, d)
        y_flat = y.reshape(-1, m)
        yvar_flat = yvar.reshape(-1, m)

        _results = []
        for x_point, y_point, yvar_point in zip(x_flat, y_flat, yvar_flat, strict=True):
            qoi_var = self.lookahead(x_point, y_point, yvar_point)
            _results.append(qoi_var)

        results = torch.tensor(_results)
        results = results.reshape(*batch_shape, n)
        return results

    def lookahead(self, x_point: torch.Tensor, y_point: torch.Tensor, yvar_point: torch.Tensor | None) -> torch.Tensor:
        """Performs a single lookahead calculation.

        Adds a single additional datapoint to the GP, and determines the QoI with the new GP.

        Args:
            x_point: (d,) The x location of the new point
            y_point: (m,) The y (target) of the new point
            yvar_point: (m,) The y_var  of the new point

        Returns:
            (,) Variance of the QoI with the lookahead GP
        """
        # if homoskedastic noise is used yvar needs to be None due to conditional_upate
        if isinstance(self.model.likelihood, GaussianLikelihood):
            yvar_point = None

        updated_model = conditional_update(
            self.model,
            X=x_point.unsqueeze(0),
            Y=y_point.unsqueeze(0),
            observation_noise=yvar_point.unsqueeze(0) if yvar_point is not None else None,
        )

        #  Calculate the QoI estimates with the fantasized model
        qoi_estimates = self.qoi(updated_model)

        # Calculate the variance of the QoI estimates
        return self.qoi.var(qoi_estimates)


def conditional_update(model: Model, X: torch.Tensor, Y: torch.Tensor, observation_noise: torch.Tensor | None) -> Model:  # noqa: N803
    """A wrapper around `BatchedMultiOutputGPyTorchModel.condition_on_observations` with a number of safety checks.

    This function adds an additional datapoint to the model, preserving the dimension of the original model. Does not
    changing any of the models hyperparameters. This is like training a new `SingleTaskGP` with all the datapoints
    (Hyperparameters are not fit in SingleTaskGP).

    Args:
        model (Model): The model to update.
        X: As per `condition_on_observations`. Shape (`*b`, n', d).

            .. Note::
                `condition_on_observations` expects this to be in the "model" space. It will not be
                transformed by the `input_transform` on the model.

        Y: As per `condition_on_observations`. Shape (`*b`, n', m).

            .. Note::
                `condition_on_observations` expects this to be in the "output/problem" space (not model space).
                It will be transformed by the `output_transform` on the model.

        observation_noise (torch.Tensor | None): Used as the `noise` argument in `condition_on_observations`. Shape
            should match `Y` (`*b`, n', m).

            .. Note::
                `condition_on_observations` expects this to be in the "model" space. It will not be
                transformed by the `output_transform` on the model.

    Return:
        - GP with the same underlying structure, including the new points, and the same original number of dimensions.

    Developer Note:
        There are different ways to create a fantasy model. The following were considered:

        - `BatchedMultiOutputGPyTorchModel.condition_on_observations`: well documented interface producing a GP of
          the same format.
        - `model.get_fantasy_model`: This is a `Gpytorch` implementation. Interface uses different notation, and
          input shape need to be manually adjusted depending on the model.
        - `model.fantasize`: This method would be very convient for our wider purpose, but its posteriors is of shape
          `(num_fantasies, batch_shape, n, m)`. Unclear if our QoI methods can handle/respect the `num_fantasies` dim.

        Revisit this at a later date.
    """
    if not isinstance(model, SingleTaskGP):
        msg = (
            f"Currently only supports models of type SingleTaskGP, received {type(model)}."
            "Currently this has only been tested for SingleTaskGP (with Homoskedastic and Fixed noise)."
            " It can likely be expanded to broader classes of model, but it is important to understand the noise"
            "expected by the model, as this can silently create incorrect GPs if incorrect. See TODO(link unit test)"
            " for details. As such, currently only GPs that have been explicitly tested are accepted."
        )
        raise NotImplementedError(msg)

    if isinstance(model.likelihood, FixedNoiseGaussianLikelihood):
        if observation_noise is None:
            msg = "Conditional update of SingleTaskGP with FixedNoise requires observation_noise to be provided."
            raise ValueError(msg)
    elif isinstance(model.likelihood, GaussianLikelihood):
        if observation_noise is not None:
            msg = (
                "Conditional update of the observation_noise is not supported for SingleTaskGPs with Homoskedastic "
                "Noise. This combination leads to inconsistent updated of the GP."
            )
            raise ValueError(msg)
    else:
        msg = (
            "Expected the model.likelihood to be either GaussianLikelihood or FixedNoiseGaussianLikelihood."
            " Recieved {model.likelihood}"
        )
        raise TypeError(msg)

    if hasattr(model, "input_transform") or hasattr(model, "outcome_transform"):
        msg = (
            "Caution: the model passed has input or outcome transforms. Need to be extremely careful that all inputs"
            "the in the correct space ('model space' vs 'problem/outcome space'). Please see docstring for more"
            " details."
        )
        warnings.warn(msg, stacklevel=5)

    if model.batch_shape != X.shape[:-2] or model.batch_shape != Y.shape[:-2]:
        msg = (
            f"Model.batch_shape is {model.batch_shape}, but X batch_shape is {X.shape[:-2]} and Y batch_shape is:"
            f" {Y.shape[:-2]}. The resultant model will not have the same batch shape as the original."
        )
        raise ValueError(msg)

    # Botorch required the model be have been run (so caches have been populated) before conditioning can be done.
    # This is the method Botrch used to do this. We run the model on a single input if required
    if model.prediction_strategy is None:
        _ = model.posterior(X=X)

    new_model = model.condition_on_observations(
        # MUST BE IN MODEL SPACE - will not be automatically transformed
        # shape require (b,n',d)
        X=X,
        # Must be in outcome space - will be automatically transformed
        # Shape require (b,n',m)
        Y=Y,
        # Must be in model space - - will not be automatically transformed
        # shape must be the same as Y
        noise=observation_noise,
    )

    return new_model


# TODO(sw 2024-11-27): there are cases where users might want to chose between closest_observational_noise and
# average_observational_noise. This should be a parameter of QoILookAhead.
def closest_observational_noise(
    new_points: torch.Tensor, train_x: torch.Tensor, train_yvar: torch.Tensor
) -> torch.Tensor:
    """Find the closest point in a training dataset, and collect its observational noise.

    Args:
        new_points: (n, d) The points to produce observational_noise. Features should be normalise to [0,1] square.
        train_x: (n',d) The points to compare similarity to. Features should be normalise to [0,1] square.
        train_yvar: (n',m) The observational variance associated with each point.

    Return:
        (n,m) Tensor with the variance for each of the new_points.


    Details:
    This function is useful for Non-Batched SingleTaskGPs because they will always have arguments of these dimension.

    Warning:
        This function is not smooth, meaning optimizers that use gradient (1st or 2nd order derivatives) such as
        L-BFGS-B will not work. The trade off is it is more robust to the effect of patterns in yvar than
        `average_observational_noise` . See Issue #213 for details.

    """
    # NOTE: Training data might be less than [0,1] because it can be standardised with preset bounds (e.g env bounds)
    # new_points could potentailly be outside of bounds if bounds were determined by the data.
    if train_x.min() < 0 or train_x.max() > 1:
        # TODO(sw 2024-11-15): We could use normalise to standardise the model here ourselves.
        msg = (
            "The models train_x data is not standardised to [0,1]. Non-standarised features will biases large"
            " features in the similarity measure used in this function."
            f" Found min={train_x.min()}, max={train_x.max()}."
        )
        warnings.warn(msg, stacklevel=8)

    # measure the distance from new_points to train_points to find the one that is most similar.
    # (new_point, distance to points)
    distances = torch.cdist(new_points, train_x, p=2)
    closest_point_idx = distances.argmin(dim=-1)
    return train_yvar[closest_point_idx]


def average_observational_noise(
    new_points: torch.Tensor,
    train_x: torch.Tensor | None,  # noqa: ARG001
    train_yvar: torch.Tensor,
) -> torch.Tensor:
    """Return the average observational noise.

    Args:
        new_points: (n, d) The points to produce observational_noise.
        train_x: (n',d). This is not used, but is kept to keep a consistent function signature
        train_yvar: (n',m) The observational variance associated with each point.

    Return:
        (n,m) Tensor with the variance for each of the new_points.

    Details:
    This function is useful for Non-Batched SingleTaskGPs because they will always have arguments of these dimension.

    Warning:
        Certain pattern of homoskedasticity cause this method to perform poorly, causing the acquisition function to
        recommend suboptimal points (as compared to `closest_observational_noise`). The trade off is derivative based
        optimisation techniques can be used. See Issue #213 for details.
    """
    return train_yvar.mean(dim=0).expand(new_points.shape[0], -1)


def _get_fantasy_observation_noise(model: SingleTaskGP, x: torch.Tensor) -> torch.Tensor:
    """Estimates the observation noise of a new point by finding the closest point in a model.

    Distance is assumed a good measure of point similarity.

    Args:
        model: The model from which to use the training data.
        x: (n,d) The points to find fantasy noise for. Batching is not supported. This should be in model space.

    Return:
        Tensor (n,m) of observation noise (variance) for each point in x.

    Todo TODO:
        - Review the implication noise picking choice on optimisation. See Issue #213 for details.
    """
    reject_if_batched_model(model)

    # Gaurenteeed to be (n,d)
    model_train_x = get_training_data_singletaskgp(model)
    # # Gaurenteeed to be (n,m) due to rejecting batched_models
    model_yvar = get_target_noise_singletaskgp(model)

    # if model uses homoskedastic noise output will need expand output from (1,m) to (n,m)
    model_yvar = model_yvar.expand(model_train_x.shape[0], -1)

    x_obs_noise = average_observational_noise(x, model_train_x, model_yvar)

    return x_obs_noise


def reject_if_batched_model(model: SingleTaskGP) -> None:
    """Helper function to reject batched model in code where they are not yet supported.

    Args:
        model: The model to check

    Return:
        Raise not yet implements in model is batched. Otherwise None.

    Details:
        botorch models can have batched training data, and or batched.

        - gp batch prediction (non-batched model):

          - train_x = (n,d) # This is a single GP
          - train_y = (n,m)
          - predicting_x = (b,n',d)
          - result will be: (b, n',m).

            - There are b seperate joint distribution (each with n points, a t targets)

        - batched gps model:

          - train_x = (b_gp,n,d)
          - train_y = (b_gp,n,m)

            - b_gp seperate GPs, where each GP gets all its own hyperparams etc trained on (n,d) point.

          - prediciton_x = (n',d)
          - result will be: (b_gp, n',m).

            - Each of the seperate b_gp gps makes its own estimate of the joint distribution.

        More details: `BoTorch batching <https://botorch.org/docs/batching>`_
    """
    if len(model.batch_shape) != 0:
        msg = (
            "batch models are not currently supported. Model has training data (*b,n,d), on (n,d) is supported."
            "See https://botorch.org/docs/batching for more details on batching"
        )
        raise NotImplementedError(msg)


@acqf_input_constructor(QoILookAhead)
def construct_inputs_qoi_look_ahead(  # noqa: D417
    model: SingleTaskGP, qoi_estimator: QoIEstimator, sampler: PosteriorSampler, **_: dict[str, Any]
) -> dict[str, Any]:
    """This is how default arguments for acquisition functions are handled in Ax/Botorch.

    Context
        When Ax.BOTORCH gets instantiated, construction arguments for the acquisition function can be provided.
        These are passed through Ax as a set of Kwargs

    Args:
        This function takes a subset of the acquisition functions `__init__()` args and can add defaults.

    Returns:
        Args for the Botorch acquisition function `__init__()` (output).

    Note:
        This functionality allows Ax to pass generic arguments without needing to know which acquisition function they
        will be passed to. Interestingly, this functionality is provided by the BoTorch package, even though it seems
        like it should be the responsibility of Ax. This issue is discussed in detail here: `GitHub discussion <https://github.com/pytorch/botorch/discussions/784>`_.
    """
    return {
        "model": model,
        "qoi_estimator": qoi_estimator,
        "sampler": sampler,
    }


@optimizer_argparse.register(QoILookAhead)
def _argparse_qoi_look_ahead(
    # NOTE: this is a bit of a non-standard implementation because we need to update params in a nested dict. Would
    #  prefer to set the key values directly here
    acqf: QoILookAhead,
    # needs to accept the variety of args it is handed, and then pick the relevant ones
    **kwargs: Any,  # noqa: ANN401
) -> dict[str, Any]:
    """Controls the default optimization parameters used.

    The following provides an overview of how passing optimization argument to `ax.TorchModelBridge.gen()`.
    Anything that is not passed here will used the default value defined by this function.

    ```python
        TorchModelBridge.gen(
            ...                                 # Other parameters there that are not relevant
            model_gen_options: {
                optimise_kwargs: {              # content of this key later used like `botorch.optim.optimize_acqf(**optimise_kwargs)`  # noqa: E501
                    options: {                  # This is a parameter of `botorch.optim.optimize_acqf`
                        with_grad: False        # Example of some optimization params that can be passed
                        ...
                        method: "L-BFSG"
                    }
                }
            }
    ```

    Rough flow of this object through the stack:
        - `ax.TorchModelBridge.gen(..., model_gen_options: dict)`
        [internally end up calling `TorchModelBridge.model` giving the below object]
        - `ax.BotorchModel.gen(..., torch_opt_config: TorchOptimConfig)`
            - Same way TorchModelBridge convert x data from ax type to tensors, the config type is converted.
        [Internally that leads to the following being called]
        - `ax.Acquisition.optimise(..., opt_options)`
            - where `opt_options = torch_opt_config.model_gen_options.optimize_kwargs`
        [Internally this leads to the following being called]
        - `botorch.optim.optimize_acqf(..., opt_options_with_defaults)`
            - `opt_options_with_defaults` is `opt_options` with defaults added.

    This function helps add the defaults in the final step

    Note:
        - This function is stored in a registry in Ax. This registery is searched (by acquisition class name)
          when the acqf func is being optimized.
        - Any variable passed from `.gen` must override the defaults specified here
        - Existing default implementations can be found here: ax.models.torch.botorch_modular.optimizer_argparse

    Args:
        acqf: Acqusiton that will be passed.
        kwargs:
            - everything passed in value of `optimizer_kwargs` in `model.gen(model_gen_options = {optimizer_kwargs ={}})` can be found with
            `kwargs[optimizer_options]`

    Returns:
        dict of args unpacked to `botorch.optim.optimize_acqf`.
            - All args are set with this, excluding the following:
                - `acq_function`,`bounds`,`q`,`inequality_constraints`,`fixed_features`,`post_processing_func`
    """  # noqa: E501
    # Start by using the default arg constructure, then adding in any kwargs that were passed in.

    # NOTE: this is an internal method, using it is an anti pattern.
    # - Ax just stores all the arge parsers in the same file as _argparse_base so they can use it directly.
    # - We can't put this function in that file, even though it belongs there.
    # Definition of _argparse_base explains the shape returned
    args = _argparse_base(acqf, **kwargs)

    # Only update with these defaults if the variable were not passin in kwargs
    optimizer_options = kwargs["optimizer_options"]
    options = optimizer_options.get("options", {})

    if "raw_samples" not in optimizer_options:
        args["raw_samples"] = 100
    if "with_grad" not in options:
        args["options"]["with_grad"] = False

    if "method" not in options:
        args["options"]["method"] = "Nelder-Mead"
        # These options are specific to using Nelder-Mead
        if "maxfev" not in options:
            args["options"]["maxfev"] = "5"
        if "retry_on_optimization_warning" not in optimizer_options:
            args["retry_on_optimization_warning"] = False

    return args
