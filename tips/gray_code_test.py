from hypothesis import given
from hypothesis.strategies import integers

from gray_code import SettingsChange
from gray_code import from_gray_code
from gray_code import gray_code_iter
from gray_code import to_gray_code


def test_convert_to_gray_code():
    assert int("00101", 2) == to_gray_code(int("00110", 2))


@given(integers(min_value=0))
def test_convert_and_back(x):
    assert x == from_gray_code(to_gray_code(x))


def test_iter():
    n = 4
    settings = lambda n, s, f: SettingsChange(num_settings=n, settings=s, flipped_bit=f)

    expected = [
        settings(n, 0, None),
        settings(n, 1, 0),
        settings(n, 3, 1),
        settings(n, 2, 0),
        settings(n, 6, 2),
        settings(n, 7, 0),
        settings(n, 5, 1),
        settings(n, 4, 0),
        settings(n, 12, 3),
        settings(n, 13, 0),
        settings(n, 15, 1),
        settings(n, 14, 0),
        settings(n, 10, 2),
        settings(n, 11, 0),
        settings(n, 9, 1),
        settings(n, 8, 0),
    ]

    assert expected == list(gray_code_iter(n))
