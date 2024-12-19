"""MarginalCDFExtrapolation calculates QoIs by estimating the behaviour of a single timestep, and extrapolating."""

# pyright: reportUnnecessaryTypeIgnoreComment=false
# %%


def q_to_qtimestep(q: float, period_len: int) -> float:
    """Convert a long term quantile to an equivalent single timestep quantile.

    Args:
        q: long term quantile being estimated.
        period_len: the number of timesteps that make up the period of interest.

    Returns:
        The equivalent quantile for a single timestep with error +-

    Note:
        This funciton exists because there were numerical concerns for this process. Having a function allows us to
        document them in tests. It is appropriately accuract for very large periods. Periods of 1e13 creates an error of
        less that 1e-3 in q (the full quantile estimate). See
        `test/qoi/test_margianl_cdf_extpolation/test_q_to_qtimestep_numerical_precision_period_increase` for details.
    """
    return q ** (1 / period_len)


def acceptable_timestep_error(q: float, period_len: int, atol: float = 0.01) -> float:
    """Maximum possible single timestep error while still producing required accuracy in period estimate.

    We often make estimates for a single timestep, and scale them to a period (many timesteps). e.g.
    `q = q_timestep^period_len`. Errors in the single timestep estimates compound. This function returns the largest
    error possible to a single timestep to still be within the required accuracy in the period estimate.

    Args:
        q: long term quantile being estimated.
        period_len: the number of timesteps that make up the period of interest.
        atol: the maximum absolute error acceptable in q

    Returns:
        The maximum absolute error acceptable in the q_timestep estimate.
    """
    # NOTE: We only look at `q + atol` because this will produce a larger error than `q - atol`
    # (because of the exponential relationship)
    acceptable_error = q_to_qtimestep(q + atol, period_len) - q_to_qtimestep(q, period_len)

    FLOAT64_RESOLUTION = 1e-15  # noqa: N806
    if acceptable_error < (FLOAT64_RESOLUTION * 10):
        msg = (
            f"The acceptable error is {acceptable_error}. This is approaching or below the limits of floating point 64"
            " precision (1e-15)."
        )
        raise ValueError(msg)
    return acceptable_error
