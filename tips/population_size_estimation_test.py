from hypothesis import given
from hypothesis.strategies import integers
from hypothesis.strategies import floats
import random
import math

from population_size_estimation import estimate_size


@given(integers(min_value=100, max_value=100000), floats(min_value=0.2, max_value=0.9))
def test_estimate_is_accurate(n, sample_frac):
    random.seed(1)
    subjects = list(range(n))
    sample = random.sample(subjects, int(n * sample_frac))

    actual = estimate_size(sample)
    # the variance is n^2 / k^2, so the stddev is n/k, and we should expect to
    # be within 3 standard deviations of the true value
    stddev = 3 * (n / len(sample))
    assert abs(actual - n) < stddev
