from typing import List
from typing import Tuple


def estimate_size(sample: List[int]) -> float:
    """
    Estimate the size of a set {1, 2, ..., n}, given a uniform random sample
    of its members without replacement.

    Args:
      - sample: a uniform random subset of the population

    Returns:
      an estimate of the population size
    """
    k = len(sample)
    m = max(sample)
    return m + m / k - 1


def size_confidence_interval(
    sample: List[int], confidence: float,
) -> Tuple[float, float]:
    """Return a confidence interval analogue of estimate_size.

    This computes the c-percent downward-biased confidence interval [0, c] of
    the size of a population given a uniform random sample of its members
    with replacement.

    Caveat: this formula uses sampling with replacement, whereas the interval
    for sampling without replacement is slightly narrower given the same
    confidence level. For practical purposes you can probably ignore the
    difference.

    Args:
      - sample: a uniform random subset of the population
      - confidence: a float 0 < c < 1

    Returns:
      A confidence interval [m, B*m] representing a (100*c)% confidence
      interval around the true underlying population size, where m is the
      maximum observation in the sample.
    """
    k = len(sample)
    m = max(sample)
    B = 1 / confidence ** (1 / k)
    return (m, B * m)
