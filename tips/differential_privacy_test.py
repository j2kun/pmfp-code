import math
import random
from collections import Counter

import numpy as np
import pytest
from hypothesis import example, given
from hypothesis import strategies as st

from tips.differential_privacy import (
    InsecureLaplaceMechanism,
    SecureLaplaceMechanism,
    next_power_of_two,
    privatize_histogram,
    sample_geometric,
)


def distributions_are_close(hist1, hist2, L2_tolerance):
    """Decides whether two sets of random samples were likely drawn from similar
    discrete distributions.

     The distributions are considered similar if the l2 distance between them
     is less than half the specified l2 tolerance t. Otherwise, if the distance
     is greater than t, they are considered dissimilar. The error probability
     is at most 4014 / (n * t^2), where n is the number of samples contained in
     one of the sets. See https://arxiv.org/abs/1611.03579 for more
     information, "Collision-based Testers are Optimal for Uniformity and
     Closeness," by Ilias Diakonikolas, Themis Gouleakis, John Peebles, and
     Eric Price.

    Arguments:
      - hist1: a dictionary whose keys are sample outcomes and whose values are
          the frequency of the outcome in the sample, for one of the two
          distributions being compared.
      - hist2: same as hist1, but for a sample from the other distribution.
      - L2_tolerance: the L2 distance between the underlying distributions, above which
          the distributions are considered dissimilar. Lower tolerance requires
          more samples to match the increased likelihood of error (even when
          given two samples from the same distribution).

    Returns True if the underlying distributions are within the given L2 distance.
    """
    n = sum(hist1.values())
    assert (n) == sum(hist2.values())

    self_collision_1 = sum(count * (count - 1) / 2 for count in hist1.values())
    self_collision_2 = sum(count * (count - 1) / 2 for count in hist2.values())
    cross_collision_count = sum(
        hist1[sample] * hist2.get(sample, 0) for sample in hist1.keys()
    )

    test_value = (
        self_collision_1 + self_collision_2 - ((n - 1) / n) * cross_collision_count
    )
    threshold = (L2_tolerance * (n - 1)) * (L2_tolerance * n) / 4.0
    return test_value < threshold


def dp_test_statistic(s1, s2, privacy_parameter):
    """Compute the epsilon-delta differential privacy test statistic.

    This checks whether the differential privacy mechanism, which outputs samples s1 and
    s2 from two neighboring databases, approximately satisfies differential privacy.
    Because the code in this Tip focuses on epsilon-DP, not epsilon-delta DP, but
    epsilon-DP is not testable in general, we resort to testing for epsilon-delta DP
    with a very small delta, hard coded in the tests that call this function.

    This statistic is from Gilbert-McMillan 2018, "Property Testing for Differential
    Privacy", Theorem 14 and Algorithm 2.
    https://arxiv.org/abs/1806.06427

    Also see a reference implementation at
    https://github.com/google/differential-privacy/blob/c2376f0daaf406e1524b462accaa9cbb548fd6d1/java/main/com/google/privacy/differentialprivacy/testing/StatisticalTestsUtil.java#L88-L137
    """
    n = sum(s1.values())
    assert n == sum(s2.values())
    return sum(
        max(0.0, (s1[k] - math.exp(privacy_parameter) * s2.get(k, 0)) / n)
        for k in s1.keys()
    )


def make_mechanisms():
    mechs = [InsecureLaplaceMechanism(), SecureLaplaceMechanism(random.SystemRandom())]
    return [(x.__class__.__name__, x) for x in mechs]


@given(
    st.floats(
        min_value=0.0,
        max_value=2**1023,
        allow_infinity=False,
        allow_nan=False,
        exclude_min=True,
    ),
)
@example(float(2**-1023))
@example(float(2**1023))
def test_next_power_of_two(x):
    output = next_power_of_two(x)
    expected = min(float(2**i) for i in range(-1022, 1024) if 2**i >= x)
    assert expected == output


def test_sample_geometric_truncation():
    assert sample_geometric(random.Random(1), 1e-100) == 1 << 64


def test_large_granularity():
    mechanism = SecureLaplaceMechanism(random.Random(1))
    output = mechanism.add_noise(11, 0.3, 1 << 40)  # results in a ganularity of 4
    # This proves that when the granularity is large, the value is first
    # rounded to a multiple of granularity before the noise is added.
    assert output[0] == 12


@pytest.mark.order(index=-1)
@pytest.mark.parametrize("name,mechanism", make_mechanisms())
def test_privatize_single_number(name, mechanism):
    number, privacy_parameter = 17, 0.5
    sample_size = 100000

    def sample(num):
        return Counter(
            [
                privatize_histogram([num], privacy_parameter, mechanism)[0]
                for _ in range(sample_size)
            ],
        )

    sample_outputs = sample(number)
    baseline = Counter(
        [
            max(0, round(x))
            for x in np.random.default_rng(1).laplace(
                number,
                1.0 / privacy_parameter,
                sample_size,
            )
        ],
    )
    assert distributions_are_close(sample_outputs, baseline, 1e-02)


@pytest.mark.order(index=-2)
@pytest.mark.parametrize("name,mechanism", make_mechanisms())
def test_privatize_single_bin_histogram(name, mechanism):
    neighboring_hists, privacy_parameter = ((17,), (18,)), math.log(3)
    hist1, hist2 = neighboring_hists
    sample_size = 700000

    def sample(hist):
        return Counter(
            [
                tuple(privatize_histogram(hist, privacy_parameter, mechanism))
                for _ in range(sample_size)
            ],
        )

    sample_hist1 = sample(hist1)
    sample_hist2 = sample(hist2)

    plot1 = [sample_hist1.get((i,), 0) for i in range(40)]
    plot2 = [sample_hist2.get((i,), 0) for i in range(40)]
    print(f"{plot1}")
    print(f"{plot2}")

    test_stat1 = dp_test_statistic(sample_hist1, sample_hist2, privacy_parameter)
    test_stat2 = dp_test_statistic(sample_hist2, sample_hist1, privacy_parameter)

    tolerance = 0.005
    assert test_stat1 < tolerance
    assert test_stat2 < tolerance


@pytest.mark.order(index=-3)
@pytest.mark.parametrize("name,mechanism", make_mechanisms())
@pytest.mark.parametrize(
    "neighboring_hists",
    [
        ((1, 2, 1, 2), (1, 2, 2, 2)),
        ((10, 15, 9, 14), (9, 15, 9, 14)),
    ],
)
def test_privatize_multi_bin_histogram(name, mechanism, neighboring_hists):
    privacy_parameter = math.log(3)
    mechanism = InsecureLaplaceMechanism()
    hist1, hist2 = neighboring_hists
    sample_size = 250000

    def sample(hist):
        return Counter(
            [
                tuple(privatize_histogram(hist, privacy_parameter, mechanism))
                for _ in range(sample_size)
            ],
        )

    sample_hist1 = sample(hist1)
    sample_hist2 = sample(hist2)

    test_stat1 = dp_test_statistic(sample_hist1, sample_hist2, privacy_parameter)
    test_stat2 = dp_test_statistic(sample_hist2, sample_hist1, privacy_parameter)

    tolerance = 0.05
    assert test_stat1 < tolerance
    assert test_stat2 < tolerance
