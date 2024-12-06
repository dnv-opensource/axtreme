# pyright: reportPrivateUsage=false

from axtreme.acquisition.qoi_look_ahead import int_to_str


def test_int_to_str():
    """Example for a simple unit test."""
    # Prepare
    int_in: int = 1
    str_out_expected: str = "1"
    # Execute
    str_out = int_to_str(int_in)
    # Assert
    assert isinstance(str_out, str)
    assert str_out == str_out_expected
