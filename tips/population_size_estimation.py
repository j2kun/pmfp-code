from typing import List


def estimate_size(sample: List[int]) -> float:
    '''
    Estimate the size of a set {1, 2, ..., n}, given a uniform random sample
    (without replacement) of its members.

    Args:
      - sample: a uniform random subset of the population

    Returns: 
      an estimate of the population size
    '''
    k = len(sample)
    m = max(sample)
    return m + m / k - 1
