"""Helper for working with scipy distibutions."""

import inspect
from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import gumbel_r, rv_continuous
from statsmodels.base.model import GenericLikelihoodModel


# How many params do we need to fit, and what are the names so we can identify them
def distribution_parameter_names_from_scipy(dist: rv_continuous) -> list[str]:
    """Collects the parameters required for a scipy rv_continous distibution.

    Withing axtreme this is used to determine what a GP should predict.

    Args:
        dist: The distibution to retrieve the parameters for.

    Returns:
        The distibutiion parameter names, IN THE ORDER they should be passed to calls to this distibution class.
        For example:

          - gumbel_r.pdf(x, loc=0, scale=1) -> order returned ['loc','scale']
          - weibull_max.pdf(x, c, loc=0, scale=1) -> order returned ['c','loc','scale']

    """
    # All rv_continous use/support 'loc' and 'scale' as detailed here: scipy.stats.rv_continuous
    parameter_names = ["loc", "scale"]
    # Collect any addition parameters the distibutuion defines.
    # This is based on how scipy itself finds the required parameters here:
    # scipy.stats.rv_continuous.__init__ in inside `_construct_argparser`
    # typically accessing priviate object is an anti-pattern, but this how scipy itself determines this
    # and ._pdf is actually user define when making new rv_contious objects
    pdf_params = list(inspect.signature(dist._pdf).parameters.keys())  # noqa: SLF001

    # Drop the first entry of pdf_params as this is always x
    # NOTE: the parameter order is important
    # This is the same order the the dist expects the data to be passed into the function signatures
    dist_params = pdf_params[1:] + parameter_names

    return dist_params


def fit_dist_with_uncertainty(
    data: ArrayLike,
    dist: rv_continuous,
) -> tuple[np.ndarray[tuple[int,], np.dtype[np.float64]], np.ndarray[tuple[int, int], np.dtype[np.float64]]]:
    """Fit a distibution, returning the parameters estimate and Gaussian uncertainty of the fit.

    Results of fitting with Maximum Likelihood give a normal distibution where

    - mean:parameter estimate
    - covariance matrix: Uncertainty in estimate (Inverse Fischer infomation matrix)

    Note:
        Parameters are returned in tha same order they appear in the fucntion calls of this call.
        For example, weibull_max.pdf(x, c, loc=0, scale=1) -> order returned ['c','loc','scale']
        This order can be extracted with the helper function `distribution_parameter_names_from_scipy`

    Args:
        data (ArrayLike): 1 dimensional set of data to fit a distibution to.
        dist (rv_continuous): the distribution to be fitted.

    Returns:
        feature mean: (n) parameter mean estimates
        covariance: (n,n) covariance matrix of the uncertainty in parameters fitted

    Todo:
        - Some distributions have convergence issues. Currently only allow gumble_r.
        - Optimisation method needs thorough testing
    """
    if type(dist) is not type(gumbel_r):
        msg = (
            "Currently only support gumbel, as the optimisation does not always converge with other distributions."
            " If using other distibution will need to perform suffecient testing then add to the allowed types in"
            " this method"
        )
        raise NotImplementedError(msg)

    # Has to return the logprob of dataset given a set of params
    # Typically this is implementented on the object which has the data stored internally
    # Using closure to conform to the function signature and still have access to the data
    # Note: they type here is a bit loose, but the effect of this is contained and this function will be overwritten.
    def loglike(params: Sequence[int | float]) -> float:
        return np.sum(dist.logpdf(data, *params))

    # GenericLikelihoodModel documentation example shows the data should also be passed here
    # (in addition to loglike function())
    model = GenericLikelihoodModel(endog=data, loglike=loglike)

    # TODO(sw): This needs testing, works to generically pass good starting points, BUT
    # Need to be careful, exact estimates have made other optimisers fail
    fitted_params = dist.fit(data)
    result = model.fit(start_params=fitted_params, disp=False)

    return result.params, result.cov_params()
