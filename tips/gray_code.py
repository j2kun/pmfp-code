from typing import Iterator
from typing import Optional
from dataclasses import dataclass


def to_gray_code(x: int) -> int:
    """Convert a counter index to its corresponding Gray code."""
    return (x >> 1) ^ x


def from_gray_code(n: int) -> int:
    """
    Convert a Gray code to its corresponding index in the sequence of
    Gray code iteration.
    """
    # This implementation is from John D. Cook, reproduced with
    # permission.
    # https://www.johndcook.com/blog/2020/09/08/inverse-gray-code/
    x, e = n, 1
    while x:
        x = n >> e
        e *= 2
        n = n ^ x
    return n


@dataclass
class SettingsChange:
    """The numbers of binary settings."""

    num_settings: int

    """
    The choice of values for the settings, as a bit mask of length
    num_settings.
    """
    settings: int

    """Which bit was flipped last."""
    flipped_bit: Optional[int]


def gray_code_iter(num_settings: int) -> Iterator[SettingsChange]:
    """
    Iterate over the range 1..2^num_settings, where each step in the
    iteration modifies only a single bit at a time.
    """
    i = 0
    next_value = to_gray_code(i)
    yield SettingsChange(
        num_settings=num_settings, settings=next_value, flipped_bit=None,
    )

    for i in range(1, 2**num_settings):
        last_value = next_value
        next_value = to_gray_code(i)
        flipped_bit = (next_value ^ last_value).bit_length() - 1
        yield SettingsChange(
            num_settings=num_settings, settings=next_value, flipped_bit=flipped_bit,
        )
