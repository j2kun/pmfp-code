import itertools
import math
import random
from collections import defaultdict

import pytest

from tips.reservoir_sample import (
    algorithm_L,
    sample_without_replacement,
    weighted_sample_with_replacement,
)


# used to assert the expected probability of each outcome
def uniform_random_without_repacement(n, k):
    return 1.0 / math.comb(n, k)


def uniform_random_with_repacement(n, k):
    return 1.0 / (n**k)


def total_variation_distance(d1, d2):
    return max(abs(d1[result] - d2[result]) for result in d1)


@pytest.mark.parametrize(
    "sampler",
    [
        sample_without_replacement,
        algorithm_L,
    ],
)
def test_reservoir_sample_probability(sampler):
    random.seed(1)
    n = 10
    sample_size = 3
    stream = list(range(n))
    expected_weight = uniform_random_without_repacement(n, sample_size)

    counter = defaultdict(int)
    experiment_count = 100000

    for _ in range(experiment_count):
        result = tuple(sorted(sampler(sample_size, stream)))
        counter[result] += 1

    assert len(counter) == int(1 / expected_weight)

    expected = {
        (i, j, k): expected_weight
        for (i, j, k) in itertools.combinations(stream, sample_size)
    }
    actual = defaultdict(
        int,
        ((result, counter[result] / experiment_count) for result in counter),
    )

    total_variation = total_variation_distance(expected, actual)
    assert total_variation < 0.01


def test_weighted_sample_with_replacement_uniform_weights():
    sampler = weighted_sample_with_replacement
    random.seed(1)
    n = 10
    sample_size = 3
    expected_weight = uniform_random_with_repacement(n, sample_size)
    stream = [(k, 1.0 / n) for k in range(n)]

    counter = defaultdict(int)
    experiment_count = 100000

    for _ in range(experiment_count):
        # Important we don't sort the tuples here, because each (unsorted)
        # tuple should occur with equal probability.
        result = tuple(sampler(sample_size, stream))
        counter[result] += 1

    assert len(counter) == int(1 / expected_weight)

    expected = {
        (i, j, k): expected_weight
        for (i, _) in stream
        for (j, _) in stream
        for (k, _) in stream
    }
    actual = defaultdict(
        int,
        ((result, counter[result] / experiment_count) for result in counter),
    )
    total_variation = total_variation_distance(expected, actual)
    assert total_variation < 0.01


def test_weighted_sample_with_replacement_point_distribution():
    sampler = weighted_sample_with_replacement
    random.seed(1)
    n = 10
    sample_size = 3
    _ = uniform_random_with_repacement(n, sample_size)

    expected = 5
    stream = [(k, 1.0 if k == expected else 0) for k in range(n)]

    experiment_count = 1000
    for _ in range(experiment_count):
        # Important we don't sort the tuples here, because each (unsorted)
        # tuple should occur with equal probability.
        result = sampler(sample_size, stream)
        assert result == [expected] * sample_size


def test_weighted_sample_with_replacement_nonuniform_weights():
    sampler = weighted_sample_with_replacement
    random.seed(1)
    n = 8
    sample_size = 3
    stream = [(k, k) for k in range(n)]
    counter = defaultdict(int)
    experiment_count = 100000

    for _ in range(experiment_count):
        result = tuple(sampler(sample_size, stream))
        counter[result] += 1

    weight_sum = sum(x[1] for x in stream)
    expected = {
        (i, j, k): i_weight * j_weight * k_weight / weight_sum**sample_size
        for (i, i_weight) in stream
        for (j, j_weight) in stream
        for (k, k_weight) in stream
    }

    actual = defaultdict(
        int,
        ((result, counter[result] / experiment_count) for result in counter),
    )

    total_variation = total_variation_distance(expected, actual)
    assert total_variation < 0.01
