from collections import defaultdict
import math
import random

import pytest

from reservoir_sample import reservoir_sample
from reservoir_sample import reservoir_sample_L

@pytest.mark.parametrize(
    "sampler", [reservoir_sample, reservoir_sample_L]
)
def test_reservoir_sample_probability(sampler):
    random.seed(1)
    n = 4
    sample_size = 2
    nums = list(range(n))
    n_choose_k = math.comb(n, sample_size)

    # 10 choose 3 == 120, implying counter will have 120 keys, and resulting
    # distribution of samples will be roughly uniform with probability 1/120.
    counter = defaultdict(int)
    experiment_count = 100000

    for _ in range(experiment_count):
        result = frozenset(sampler(sample_size, nums))
        counter[result] += 1

    assert len(counter) == n_choose_k

    expected_uniform_weight = 1.0 / n_choose_k
    distribution = [counter[result] / experiment_count for result in counter]
    for (s, k) in counter.items():
        print(f"{set(s)}: {k}")
    deviations = [abs(x - expected_uniform_weight) for x in distribution]
    total_variation = max(deviations)

    assert total_variation < 0.01, f"Distribution was {distribution}, deviations={deviations}"
