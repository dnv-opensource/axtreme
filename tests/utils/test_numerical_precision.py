import pytest
import torch

from axtreme.utils.numerical_precision import maximal_representation_error_between_0_1


@pytest.fixture(params=[torch.float32, torch.float64])
def dtype(request: pytest.FixtureRequest) -> torch.dtype:
    return request.param


def test_maximal_representation_error_between_0_1_smaller_error_cause_no_rounding(dtype: torch.dtype):
    # representation should be insenstive to change of less than maximal representaion error
    x = torch.tensor(0.6, dtype=dtype)

    assert x + maximal_representation_error_between_0_1(x.dtype) * 0.9 == x
    assert x + maximal_representation_error_between_0_1(x.dtype) * -0.9 == x


def test_maximal_representation_error_between_0_1_larger_error_cause_rounding(dtype: torch.dtype):
    """Larger eerors are expected to cause rounding to the nex smallest representable number."""
    x = torch.tensor(0.6, dtype=dtype)

    # eps is the smallest step from [1,2). As we are in the range [.5,1), this shrink by half
    # This is the same as 2^-(mantisa + 1) as detailed in maximal_representation_error_between_0_1
    smallest_rep = torch.finfo(x.dtype).eps / 2

    # changing by maximal_representation_error will cause rounding
    assert x + maximal_representation_error_between_0_1(x.dtype) * 1.1 == x + smallest_rep
    assert x + maximal_representation_error_between_0_1(x.dtype) * -1.1 == x - smallest_rep


def test_maximal_representation_error_between_0_1_exact_error(dtype: torch.dtype):
    """The rounding behaviour when the error is exactly the maximal representation error.

    Behaviour here is a bit surpising. It doesn't round up, instead round to the binary number that is even in binary
    (last mantisa is 1).
    """

    # we pick a number that can be represented perfectly in binary. We then know the final binary digit is 0.
    x = torch.tensor(0.75, dtype=dtype)

    # Both exact error are expect to round back to the original number
    assert x + maximal_representation_error_between_0_1(x.dtype) == x
    assert x - maximal_representation_error_between_0_1(x.dtype) == x

    # eps is the smallest step from [1,2). As we are in the range [.5,1), this shrink by half
    # This is the same as 2^-(mantisa + 1) as detailed in maximal_representation_error_between_0_1
    smallest_rep = torch.finfo(x.dtype).eps / 2

    # The binary number above is odd, so this should round down to our original number
    assert x + smallest_rep - maximal_representation_error_between_0_1(x.dtype) == x

    # The binary number below is odd, so this should round up to our original number
    assert x - smallest_rep + maximal_representation_error_between_0_1(x.dtype) == x
