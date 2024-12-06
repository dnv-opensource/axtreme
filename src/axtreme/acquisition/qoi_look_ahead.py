"""Utility functions for integer to string conversion."""

import logging

__ALL__ = ["int_to_str"]

logger = logging.getLogger(__name__)


def int_to_str(value: int) -> str:
    """Convert integer to string.

    This function converts an integer to its string representation.

    Parameters
    ----------
    value : int
        the integer to be converted to string

    Returns
    -------
    str
        the resulting string
    """
    logger.debug(f"function int_to_str() in acquisition called with argument {value}")
    return str(value)
