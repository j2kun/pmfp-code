'''An implementation of the Laplacian mechanism for privately releasing a
histogram where each underlying user contributes to a single bin.'''

from abc import ABC, abstractmethod
from struct import pack
from struct import unpack
from typing import List
import math

import numpy as np
import sys


Histogram = List[int]

EXPONENT_MASK = 0x7ff0000000000000
MANTISSA_MASK = 0x000fffffffffffff
EXPONENT_ONE = 0x0010000000000000
MAX_EXPONENT_BITS = unpack('>Q', pack('>d', sys.float_info.max))[0] & EXPONENT_MASK


def next_power_of_two(x: float) -> float:
    assert x > 0 and x != float('inf'), (
        f"Expecting a finite, positive number, got {x}")

    bits = unpack('>Q', pack('>d', x))[0]

    # For a finite positive IEEE float, x is a power of 2 if and only if its
    # mantissa is zero.
    if bits & MANTISSA_MASK == 0:
        return x

    exponent_bits = bits & EXPONENT_MASK

    assert exponent_bits < MAX_EXPONENT_BITS, f"Expecting a number less than 2^1023, got {x}"

    # Add 1 to the exponent bits to get the next power of 2, resetting mantissa
    # to zero.
    rounded = exponent_bits + EXPONENT_ONE
    return unpack('>d', pack('>Q', rounded))[0]


def sample_geometric(rng, exponent):
    """ Returns a sample drawn from the geometric distribution of parameter p =
    1 - e^-exponent, i.e., the number of Bernoulli trials until the first
    success where the success probability is 1 - e^-exponent.
    """

    max_value = 1 << 64
    # Return truncated sample in the case that the sample exceeds the max value.
    if (rng.random() > -1.0 * math.expm1(-1.0 * exponent * max_value)):
        return max_value

    # Perform a binary search for the sample in the interval from 1 to max long. Each iteration
    # splits the interval in two and randomly keeps either the left or the right subinterval
    # depending on the respective probability of the sample being contained in them. The search
    # ends once the interval only contains a single sample.
    left = 0  # exclusive
    right = max_value  # inclusive

    while (left + 1 < right):
        # Compute a midpoint that divides the probability mass of the current
        # interval approximately evenly between the left and right subinterval.
        # The resulting midpoint will be less or equal to the arithmetic mean
        # of the interval. This reduces the expected number of iterations of
        # the binary search compared to a search that uses the arithmetic mean
        # as a midpoint. The speed up is more pronounced, the higher the
        # success probability p is.
        mid = math.ceil(
            left - (math.log(0.5) + math.log1p(math.exp(exponent * (left - right)))) / exponent
        )

        # Ensure that mid is contained in the search interval. This is a
        # safeguard to account for potential mathematical inaccuracies due to
        # finite precision arithmetic.
        mid = min(max(mid, left + 1), right - 1)

        # Probability that the sample is at most mid, i.e.,
        #
        #    q = Pr[X ≤ mid | left < X ≤ right]
        #
        # where X denotes the sample. The value of q should be approximately
        # one half.
        q = math.expm1(exponent * (left - mid)) / math.expm1(exponent * (left - right))
        if (rng.random() <= q):
            right = mid
        else:
            left = mid

    return right


def sample_two_sided_geometric(rng, exponent):
    """Returns a sample drawn from a geometric distribution that is mirrored at
    0. The non-negative part of the distribution's PDF matches the PDF of a
    geometric distribution of parameter p = 1 - e^-exponent that is
    shifted to the left by 1 and scaled accordingly.
    """
    geometric_sample = 0
    sign = False

    # Keep a sample of 0 only if the sign is positive. Otherwise, the
    # probability of 0 would be twice as high as it should be.
    while (geometric_sample == 0 and not sign):
        geometric_sample = sample_geometric(rng, exponent) - 1
        sign = rng.random() < 0.5

    return geometric_sample if sign else -geometric_sample


class LaplaceMechanism(ABC):
    '''An interface for a random number generator that adds Laplacian noise to
    a single number, generated from a 0-mean discrete Laplacian distribution,
    i.e., with density h(y) proportional to exp(−|y|/scale), where scale is a
    parameter.

    Because the scale and the details of the mechanism depend on the privacy
    parameter, the method does not accept scale directly, but must instead
    derive the appropriate scale from the privacy parameters.

    Args:
     - value: the integer value to add noise to
     - privacy_parameter: the epsilon in differential privacy
     - sensitivity: the maximum value a single user can influence the number
         being masked.

    Returns:
      A masked version of the number that satisfies epsilon differential
      privacy.
    '''
    @abstractmethod
    def add_noise(self, value: int, privacy_parameter: float, sensitivity: float) -> int:
        ...


class InsecureLaplaceMechanism(LaplaceMechanism):
    def add_noise(self, value: int, privacy_parameter: float, sensitivity: float) -> int:
        scale = sensitivity / privacy_parameter
        return value + round(np.random.default_rng().laplace(0, scale, 1)[0])


class SecureLaplaceMechanism(LaplaceMechanism):
    '''A secure random generator for use in Differential Privacy.

    Follows the outline of
    https://github.com/google/differential-privacy/blob/main/common_docs/Secure_Noise_Generation.pdf
    and the reference implementation at
    https://github.com/google/differential-privacy/blob/c2376f0daaf406e1524b462accaa9cbb548fd6d1/java/main/com/google/privacy/differentialprivacy/LaplaceNoise.java

    For simplicity, we only support adding noise to integers. Adding noise to
    float values requires additional rounding of the input value to be a
    multiple of the granularity.
    '''
    GRANULARITY_PARAM = float(1 << 40)

    def __init__(self, rng):
        self.rng = rng

    def add_noise(self, value: int, privacy_parameter: float, sensitivity: float) -> int:
        eps = privacy_parameter
        granularity = next_power_of_two((1 / eps) / self.GRANULARITY_PARAM)
        noise = sample_two_sided_geometric(
            self.rng, granularity * eps / (sensitivity + granularity))

        # note this is where we rely on `value` being an integer.
        if granularity <= 1:
            return round(noise * granularity)
        else:
            granularity = int(granularity)
            rounded = granularity * (
                int(value / granularity) + round((value % granularity) / granularity)
            )
            return rounded + noise * granularity


def privatize_histogram(
        hist: Histogram,
        privacy_parameter: float,
        laplace: LaplaceMechanism):
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
    noisy_hist = [
        laplace.add_noise(bin_value, privacy_parameter, sensitivity=1)
        for bin_value in hist
    ]
    return [max(0, val) for val in noisy_hist]


if __name__ == "__main__":
    import cProfile
    cProfile.run(
        'for i in range(100000): '
        'privatize_histogram((17,), math.log(3), SecureLaplaceMechanism(random.SystemRandom()))')
