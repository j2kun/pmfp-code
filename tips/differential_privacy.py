'''An implementation of the Laplacian mechanism for privately releasing a
histogram where each underlying user contributes to a single bin.'''

from typing import List

from abc import ABC, abstractmethod


Histogram = List[int]


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
