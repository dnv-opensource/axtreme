#!/usr/bin/env python3
# ruff: noqa: PGH004
# ruff: noqa
# type: ignore
"""Created on Tue Apr  8 10:25:07 2025.

@author: grams
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as scst


def omega_to_k_rpc205(om, h):
    g = 9.81
    k0 = om**2 / g
    if np.isinf(h):
        return k0
    lam0 = 2 * np.pi / k0
    kh = k0 * h
    xi0 = np.sqrt(kh) * (1 + kh / 6 + kh * kh / 30)

    return 2 * np.pi / (lam0 * np.tanh(xi0))


def Tm02_from_Tp_gamma(Tp, gamma):
    return (0.6673 + 0.05037 * gamma - 0.006230 * gamma**2 + 0.0003341 * gamma**3) * Tp


def Tm01_from_Tp_gamma(Tp, gamma):
    return (0.7303 + 0.04936 * gamma - 0.006556 * gamma**2 + 0.0003610 * gamma**3) * Tp


def gamma_rpc205(Hs, Tp):
    # Find gamma from Hs, Tp using RP-C205

    R = np.atleast_1d(Tp / np.sqrt(Hs))  # noqa: N806

    gamma = np.exp(5.75 - 1.15 * R)
    gamma[np.nonzero(R <= 3.6)] = 5.0  # noqa: PLR2004
    gamma[np.nonzero(R >= 5)] = 1.0  # noqa: PLR2004

    if np.isscalar(Hs) and np.isscalar(Tp):
        gamma = gamma.item()

    return gamma


def sample_from_f_tp(hs, prms, seed):
    a1, a2, a3, b1, b2, b3 = prms

    def mu(h, a1, a2, a3):
        return a1 + a2 * h**a3

    def sig(h, b1, b2, b3):
        return np.sqrt(b1 + b2 * np.exp(-b3 * h))

    rng = np.random.default_rng(seed)

    tp = rng.lognormal(mean=mu(hs, a1, a2, a3), sigma=sig(hs, b1, b2, b3))

    return tp


def sample_from_f_hs(prms, size, seed):
    c, loc, scale = prms
    return scst.weibull_min.rvs(c, loc=loc, scale=scale, size=size, random_state=seed)


def pdf_hs_tp(hs, tp, weib_prms, lognorm_prms):
    a1, a2, a3, b1, b2, b3 = lognorm_prms
    c, loc, scale = weib_prms

    def mu(h, a1, a2, a3):
        return a1 + a2 * h**a3

    def sig(h, b1, b2, b3):
        return np.sqrt(b1 + b2 * np.exp(-b3 * h))

    pdf_hs = scst.weibull_min.pdf(hs, c, loc=loc, scale=scale)

    pdf_tp = scst.lognorm.pdf(tp, s=sig(hs, b1, b2, b3), loc=0.0, scale=np.exp(mu(hs, a1, a2, a3)))

    return pdf_hs * pdf_tp


def sample_seastates(n_ss, weib_prms, lognorm_prms, seed=None, hslim=0):
    hs = sample_from_f_hs(weib_prms, n_ss, seed)
    # sample only above hslim
    m = hs >= hslim
    hs = hs[m]
    tp = sample_from_f_tp(hs, lognorm_prms, seed)

    return hs, tp


# %%
if __name__ == "__main__":
    from wave_distributions import ForristallCrest

    weib_prms = (1.2550, 0.5026, 1.9536)
    lognorm_prms = (1.557, 0.403, 0.408, 0.005, 0.137, 0.454)
    h = 110  # water depth 110 m

    n_ss = 2922 * 10**4
    Hs, Tp = sample_seastates(n_ss, weib_prms, lognorm_prms, hslim=7.5)
    _ = plt.figure()
    _ = plt.plot(Hs, Tp, ".")

    # find jonswap gamma, Tm01 and Tm02

    gamma = gamma_rpc205(Hs, Tp)
    Tm01 = Tm01_from_Tp_gamma(Tp, gamma)
    Tm02 = Tm02_from_Tp_gamma(Tp, gamma)
    km01 = omega_to_k_rpc205(2 * np.pi / Tm01, h)

    Nw = 3600 * 3 / Tm02  # number of waves/crests in 3-hours

    F = ForristallCrest(Hs, Tm01, km01, h)

    # sample maximum crests in 3-hours
    c_max3hr = F.rvs_max(Nw).ravel()

    _ = plt.figure()
    _ = plt.hist(c_max3hr, bins=100)

    # take out the 100-year return value
    R = 100
    nR = 2922 * R  # number of c_max in 100-year

    # the value in the sorted array that correspond to the 1 - 1/nR quantile
    i_100 = n_ss // nR + 1
    c_100 = np.sort(c_max3hr)[-i_100]

    # alternative method
    # x = np.zeros(n_ss)  # noqa: ERA001
    # x[:c_max3hr.size] = c_max3hr  # noqa: ERA001
    # c_100_alt = np.quantile(x, 1 - 1/nR)  # noqa: ERA001

# %%
