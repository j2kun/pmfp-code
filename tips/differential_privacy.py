'''An implementation of the Laplacian mechanism for privately releasing a
histogram where each underlying user contributes to a single bin.'''

from typing import List
from abc import ABC, abstractmethod

import sys
import bitstring


Histogram = List[int]

EXPONENT_MASK = bitstring.pack('>Q', 0x7ff0000000000000)
MANTISSA_MASK = bitstring.pack('>Q', 0x000fffffffffffff)
EXPONENT_ONE = 0x0010000000000000


def next_power_of_two(x: float) -> float:
    assert x > 0 and x != float('inf'), (
        f"Expecting a finite, positive number, got {x}")

    bits = bitstring.pack('>d', x)

    # For a finite positive IEEE float, x is a power of 2 if and only if its
    # mantissa is zero.
    if (bits & MANTISSA_MASK).int == 0:
        return x

    exponent_bits = bits & EXPONENT_MASK
    max_exponent_bits = bitstring.pack('>d', sys.float_info.max) & EXPONENT_MASK

    assert exponent_bits.int < max_exponent_bits.int, (
        f"Expecting a number less than 2^1023, got {x}")

    # Add 1 to the exponent bits to get the next power of 2, resetting mantissa
    # to zero.
    rounded = exponent_bits.int + EXPONENT_ONE
    return bitstring.pack('>Q', rounded).float


class LaplaceGenerator(ABC):
    '''An interface for a random number generator that generates 0-mean
    Laplacian noise, i.e., with density h(y) proportional to exp(−|y|/scale),
    where scale is a parameter.
    '''
    @abstractmethod
    def sample(self, scale) -> float:
        ...


def privatize_histogram(
        hist: Histogram,
        privacy_parameter: float,
        rng: LaplaceGenerator):
    '''Privatize a histogram for public release.

    This implementation relies on the following properties:

     - Each user contributes at most a count of one to at most one bin.
     - The categories (bin definitions) of the histogram are fixed and public.

    If these are satisfied, then the mechanism satsifies epsilon-differential
    privacy for epsilon = `privacy_parameter`, meaning that if any individual
    is removed from the dataset, the probability of this mechanism producing a
    different output is at most exp(privacy_parameter). This limits the amount
    of information any attacker (using any method or extra side data) can learn
    about one individual in the dataset.
    '''
    # TODO: get a secure RNG
    laplace_scale = 1 / privacy_parameter
    noisy_hist = [bin_value + rng.sample(laplace_scale) for bin_value in hist]
    rounded_noisy_hist = [max(0, round(val)) for val in noisy_hist]
    return rounded_noisy_hist
