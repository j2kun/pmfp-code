import random

from hypothesis import given, settings
from hypothesis.strategies import floats, integers

from tips.population_size_estimation import estimate_size, size_confidence_interval


@given(integers(min_value=100, max_value=10000), floats(min_value=0.3, max_value=0.9))
@settings(deadline=1000)
def test_estimate_is_accurate(n, sample_frac):
    failures = 0
    counterexamples = []
    subjects = list(range(n))
    attempts = 20

    for i in range(attempts):
        sample = random.sample(subjects, int(n * sample_frac))

        low, high = size_confidence_interval(sample, 0.001)
        success = low <= n <= high
        if not success:
            failures += 1
            counterexamples.append([n, "confidence_interval", low, high])

        actual = estimate_size(sample)
        # the variance is n^2 / k^2, so the stddev is n/k, and we should expect to
        # be within 4 standard deviations of the true value.
        stddev = n / len(sample)
        threshold = 4 * stddev
        success = abs(actual - n) < threshold

        if not success:
            failures += 1
            counterexamples.append([n, "point_estimate", actual, threshold])

    if counterexamples:
        print(counterexamples)
        assert failures / attempts <= 0.1
