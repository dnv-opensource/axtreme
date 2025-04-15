#!/usr/bin/env python3
"""Various distributions for wave crests and wave heights.

Created on Tue Nov 26 11:11:37 2019

@author: oding
"""

# mypy: ignore-errors
# ruff: noqa: ANN001, ANN002, ANN003, ANN101, ANN201, ANN202, ANN203, D103, D104, D105, D200, D400, N802, N803, PLR2004, PLR2005, S307, G010, PGH003, TRY002
# pyright: reportMissingImports=false, reportMissingTypeArgument=false, reportUnnecessaryComparison=false, reportAttributeAccessIssue=false
# pyright: reportUnknownParameterType=false, reportCallIssue=false, reportAssignmentType=false
import numpy as np
from scipy.stats import weibull_min


def to_2d_n1(x: float | np.ndarray):
    if x is None:
        return x

    if np.isscalar(x):
        x = np.atleast_2d(x)
    elif x.ndim != 2:  # if ndim == 2, keep original shape
        x = np.atleast_2d(x).T

    if x.ndim > 2:
        raise Exception("Input must be scalar, 1d-array or 2d-array.")

    return x


def to_2d_1n(x: float | np.ndarray):
    if x is None:
        return x

    if np.isscalar(x) or x.ndim != 2:
        x = np.atleast_2d(x)

    if x.ndim > 2:
        raise Exception("Input must be scalar, 1d-array or 2d-array .")

    return x


class ProbabilityDistribution:  # noqa: D101
    def cdf(self, x: np.ndarray):
        """Cumulative distribution function.

        Parameters
        ----------
        x : ndarray
            Values for which to calculate the cdf.

        Returns:
        -------
        cdf in the points x.

        """
        return 1 - self.sf(to_2d_1n(x))

    def ppf(self, q: np.ndarray):
        """Percent point function (inverse of `cdf`).

        Parameters
        ----------
        q : ndarray
            Values for which to calculate the ppf.

        Returns:
        -------
        ppf in the points q.
        """
        return self.isf(1 - to_2d_1n(q))

    def rvs(self, size: int = 1, seed: int | None = None):
        """Random samples from distribution.

        Parameters
        ----------
        size : integer, optional
            Number of random samples to return. The default is 1.

        Returns:
        -------
        ndarray with random samples.

        """
        rng = np.random.default_rng(seed)
        u = rng.uniform(size=size)
        return self.isf(1 - u)

    def sf_max(self, n: float, x: np.ndarray):
        """Sf for the distribution of n samples from the distribution.

        Parameters
        ----------
        n : float
            Number of maxima.
        x : ndarray
            Values for which to calculate the sf_max.

        Returns:
        -------
        sf_max in the points x.
        """
        return 1 - (1 - self.sf(to_2d_1n(x))) ** to_2d_n1(n)

    def cdf_max(self, n: float, x: np.ndarray):
        """Cdf for the distribution of n samples from the distribution.

        Parameters
        ----------
        n : float
            Number of maxima.
        x : ndarray
            Values for which to calculate the cdf_max.

        Returns:
        -------
        cdf_max in the points x.
        """
        return self.cdf(to_2d_1n(x)) ** to_2d_n1(n)

    def logcdf_max(self, n: float, x: np.ndarray):
        """Logcdf for the distribution of n samples from the distribution.

        Parameters
        ----------
        n : float
            Number of maxima.
        x : ndarray
            Values for which to calculate the logcdf_max.

        Returns:
        -------
        logcdf_max in the points x.
        """
        return to_2d_n1(n) * self.logcdf(x)

    def isf_max(self, n: float, q: np.ndarray):
        """Inverse sf for the distribution maximum of of n samples from the distribution.

        Parameters
        ----------
        n : float
            Number of maxima.
        q : ndarray
            Values for which to calculate the isf_max.

        Returns:
        -------
        isf_max in the points q.
        """
        return self.isf(1 - (1 - q) ** (1 / n))

    def pdf_max(self, n: float, x: np.ndarray):
        """Probability density function for the distribution maximum of n samples from the underlying distribution.

        Parameters
        ----------
        n : float
            Number of maxima.
        x : ndarray
            Values for which to calculate the pdf.

        Returns:
        -------
        pdf_max in the points x.

        """
        return np.squeeze(n * (self.cdf(x) ** (n - 1)) * self.pdf(x))

    def rvs_max(self, n: float, size: float = 1, seed: int | None = None):
        """Random samples from the distribution of maximum of n samples.

        Parameters
        ----------
        n : float
            Number of maxima.
        size : integer, optional
            Number of random samples to return. The default is 1.

        Returns:
        -------
        ndarray with random samples.

        """
        self.rng = np.random.default_rng(seed)

        n = to_2d_n1(n)

        u = self.rng.uniform(size=(self.shape[0], size))
        r = np.asarray(self.isf_max(n, u))

        return r


class Weibull(ProbabilityDistribution):
    "Base class for all distributions that can be expressed as Weibull."

    def __init__(self, alpha: float, beta: float, gamma: float = 0) -> None:
        """Constructor.

        Parameters
        ----------
        alpha : float
            Weibull scale-parameter.
        beta : float
            Weibull shape-parameter.
        gamma : float, optional
            Weibull location-parameter. Default is 0.

        """
        # set up for broadcasting
        self.alpha = to_2d_n1(alpha)
        self.beta = to_2d_n1(beta)
        self.gamma = to_2d_n1(gamma)

        self._weib_min = weibull_min(c=self.beta, loc=self.gamma, scale=self.alpha)

        self.shape = np.broadcast_shapes(self.alpha.shape, self.beta.shape, self.gamma.shape)
        assert len(self.shape) == 2 and self.shape[1] == 1, "Error in shapes of input arrays."  # noqa: PT018

    def pdf(self, x: np.ndarray):
        """Probability density function.

        Parameters
        ----------
        x : ndarray
            Values for which to calculate the pdf.

        Returns:
        -------
        pdf in the points x.

        """
        return self._weib_min.pdf(to_2d_1n(x))

    def cdf(self, x: np.ndarray):
        """Cumulative distribution function.

        Parameters
        ----------
        x : ndarray
            Values for which to calculate the cdf.

        Returns:
        -------
        cdf in the points x.

        """
        return self._weib_min.cdf(to_2d_1n(x))

    def logcdf(self, x: np.ndarray):
        """Log of the cumulative distribution function.

        Parameters
        ----------
        x : ndarray
            Values for which to calculate the logcdf.

        Returns:
        -------
        logcdf in the points x.

        """
        return self._weib_min.logcdf(to_2d_1n(x))

    def sf(self, x: np.ndarray):
        """Survival function/exceedance probability/1 - cdf.

        Parameters
        ----------
        x : ndarray
            Values for which to calculate the sf.

        Returns:
        -------
        sf in the points x.

        """
        return self._weib_min.sf(to_2d_1n(x))

    def isf(self, q: np.ndarray):
        """Inverse survival function/exceedance probability/1 - cdf.

        Parameters
        ----------
        q : ndarray
            Values for which to calculate the isf.

        Returns:
        -------
        isf in the points q.

        """
        return self._weib_min.isf(to_2d_1n(q))

    def rvs(self, size: int = 1, random_state: int | None = None):
        """Random samples from distribution.

        Parameters
        ----------
        size : integer, optional
            Number of random samples to return (for each parameter value). The default is 1.

        Returns:
        -------
        ndarray with random samples.

        """
        return self._weib_min.rvs((self.alpha.size, size), random_state=random_state)


class ForristallCrest(Weibull):
    """Class for Forristall (2000) crest distribution."""

    def __init__(self, hs: float, t01: float, k01: float, h: float) -> None:
        """Constructor.

        Parameters
        ----------
        hs : float
            Significant wave height.
        t01 : float
            Wave period t01
        k01 : float
            Wavenumber k01
        h : float
            Water depth

        """
        dim = 3
        alpha, beta = forristall2000_prms(hs, t01, k01, h, dim)
        self.hs = hs
        super().__init__(alpha=alpha, beta=beta, gamma=0)


def forristall2000_prms(hs: float, t01: float, k01: float, h: float, dim: int):
    """Find Weibull parameters for the given sea state parameters.

    Parameters
    ----------
    hs : float
        Significant wave height.
    t01 : float
        Wave period Tm01.
    k01 : float
        Wavenumber k01 (corresponding to Tm01). The default is None.
    h : float
        Water depth. The default is None.
    dim : integer
        Number of dimensions (2 [short-crested] or 3 [short-crested]).

    """
    g = 9.81

    s = 2 * np.pi * hs / (g * t01 * t01)
    ur = np.where(np.isinf(h), 0.0, hs / (k01**2 * h**3))

    if dim == 2:
        alpha = hs * (1 / np.sqrt(8) + 0.2892 * s + 0.1060 * ur)
        beta = 2 - 2.1597 * s + 0.0968 * ur * ur
    elif dim == 3:
        alpha = hs * (1 / np.sqrt(8) + 0.2568 * s + 0.08 * ur)
        beta = 2 - 1.7912 * s - 0.5302 * ur + 0.2824 * ur * ur

    return alpha, beta
