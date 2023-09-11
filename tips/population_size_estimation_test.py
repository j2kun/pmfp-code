import random
from itertools import product

import pytest

from tips.population_size_estimation import estimate_size, size_confidence_interval


@pytest.mark.parametrize(
    "n, sample_frac",
    list(product([200, 1000, 10000], [0.1, 0.3, 0.5, 0.7, 0.9])),
)
def test_estimate_is_accurate(n, sample_frac):
    interval_failures = 0
    point_failures = 0
    counterexamples = []
    subjects = list(range(n))
    attempts = 300
    confidence = 0.95

    for i in range(attempts):
        sample = random.sample(subjects, int(n * sample_frac))

        low, high = size_confidence_interval(sample, confidence)
        success = low <= n <= high
        if not success:
            interval_failures += 1
            counterexamples.append(["confidence_interval", low, n, high])

        actual = estimate_size(sample)
        # the variance is n^2 / k^2, so the stddev is n/k, and we should expect to
        # be within 4 standard deviations of the true value.
        stddev = n / len(sample)
        threshold = 4 * stddev
        success = abs(actual - n) < threshold

        if not success:
            point_failures += 1
            counterexamples.append(
                ["point_estimate", n, actual, threshold, abs(n - actual)],
            )

    if counterexamples:
        print(counterexamples)
        assert interval_failures / attempts <= confidence
        assert point_failures / attempts <= confidence
