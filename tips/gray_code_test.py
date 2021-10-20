from hypothesis import given
from hypothesis.strategies import integers

from gray_code import from_gray_code
from gray_code import to_gray_code


def test_convert_to_gray_code():
    assert int('00101', 2) == to_gray_code(int('00110', 2))


@given(integers(min_value = 0))
def test_convert_and_back(x):
    assert x == from_gray_code(to_gray_code(x))
