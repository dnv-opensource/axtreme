"""Helpers and documentation for understanding the impacts of numerical precision."""

import torch


def maximal_representation_error_between_0_1(dtype: torch.dtype) -> float:
    """Maximal amount a float can be wrong by, for numbers within the range [0,1).

    NOTE: This is calculated for the range [.5,1). Errors will be smaller for ranges < .5.

    Args:
        dtype: The dtype to calculate the maximal representation error for.

    Returns:
        The maximal representation error for numbers between 0 and 1.

    Details:
        Mantisa (significant bits) with `x` explicity bits internally has 1 implict bit added. Effectively this become
        1.[x bits] (in binary).

            - e.g float32 has 24 bits of precision but 23 bits of mantisa

        Epsilon, the smallest **representable** number where 1+eps != 1 can be calculated as `2^-x`

            - e.g float32: 2^-23 = 1.19e-7

        The largest representable number less than 1 is calculated as: `1 - 2^-(x + 1)`:

            - e.g float32: 1 - 2^-24 = 0.999999940395355225

        As such the smallest representable number between [.5,1) is:  `2^-(x + 1)`

            - e.g float32:  2^-24 = 5.96e-08

        Value smaller than the smallest representable number will be rounded to the nearest representable number. As
        such the largest representation error for number between [.5,1) is `2^-(x + 1)/2` or `2^-(x_2)`

            - e.g float32:  2^-(23+2) = 2.9802322387695312e-08

        Resoltuion and Precisions:
        - TODO(sw): fill out exactly how these work. These number are ignored for now.

        Additional information:

            - https://en.wikipedia.org/wiki/Single-precision_floating-point_format
    """
    mantisa_bits = {
        torch.float16: 10,
        torch.float32: 23,
        torch.float64: 52,
    }
    return 2 ** -(mantisa_bits[dtype] + 1 + 1)
