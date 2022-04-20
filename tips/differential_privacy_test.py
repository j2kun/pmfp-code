import math
from collections import Counter

from hypothesis import strategies as st
from hypothesis import given
from hypothesis import example
import numpy as np

from differential_privacy import InsecureLaplaceMechanism
from differential_privacy import privatize_histogram
from differential_privacy import next_power_of_two



def distributions_are_close(hist1, hist2, L2_tolerance):
    '''
     Decides whether two sets of random samples were likely drawn from similar
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
    '''
    n = sum(hist1.values())
    assert(n) == sum(hist2.values())

    self_collision_1 = sum(count * (count - 1) / 2 for count in hist1.values())
    self_collision_2 = sum(count * (count - 1) / 2 for count in hist2.values())
    cross_collision_count = sum(hist1[sample] * hist2.get(sample, 0) for sample in hist1.keys())

    test_value = (self_collision_1 + self_collision_2 - ((n - 1) / n) * cross_collision_count)
    threshold = (L2_tolerance * (n - 1)) * (L2_tolerance * n) / 4.0
    return test_value < threshold


@st.composite
def neighboring_histograms(draw, elements, min_size=1, max_size=10):
    """Generate two neighboring histograms, i.e., two histograms that differ by
    1 in a single bin."""
    hist = draw(st.lists(elements, min_size=min_size, max_size=max_size))
    hist1 = tuple(hist)

    index = draw(st.integers(min_value=0, max_value=len(hist) - 1))
    change = draw(st.sampled_from([-1, 1]))
    hist[index] += change
    hist2 = tuple(hist)

    return (hist1, hist2)


@given(st.floats(
    min_value=0.0, max_value=2**1023, allow_infinity=False, allow_nan=False, exclude_min=True
))
@example(float(2**-1023))
@example(float(2**1023))
def test_next_power_of_two(x):
    output = next_power_of_two(x)
    expected = min(float(2**i) for i in range(-1022, 1024) if 2**i >= x)
    assert expected == output


def test_privatize_single_number():
    number, privacy_parameter = 17, 0.5

    rng = InsecureLaplaceMechanism()
    sample_size = 100000

    def sample(num):
        return Counter(
            [privatize_histogram([num], privacy_parameter, rng)[0]
             for _ in range(sample_size)])

    sample_outputs = sample(number)
    baseline = Counter([max(0, round(x)) for x in np.random.default_rng().laplace(
        17, 1.0 / privacy_parameter, sample_size)])
    assert distributions_are_close(sample_outputs, baseline, 1e-02)


# @given(neighboring_histograms(st.integers(min_value=0, max_value=100)),
#        st.floats(min_value=0.001, max_value=0.5, allow_infinity=False,
#                  allow_nan=False),
#         )
# @example(((1, 2, 1, 2), (1, 2, 2, 2)), 0.5)
def test_privatize_histogram():  # neighboring_hists, privacy_parameter):
    neighboring_hists, privacy_parameter = ((17,), (18,)), math.log(3)

    rng = InsecureLaplaceMechanism()
    hist1, hist2 = neighboring_hists
    sample_size = 200000

    def sample(hist):
        return Counter(
            [tuple(privatize_histogram(hist, privacy_parameter, rng))
             for _ in range(sample_size)])

    sample_hist1 = sample(hist1)
    sample_hist2 = sample(hist2)

    plot1 = [sample_hist1.get((i,), 0) for i in range(40)]
    plot2 = [sample_hist2.get((i,), 0) for i in range(40)]
    print(f'{plot1}')
    print(f'{plot2}')

    def dp_test_statistic(s1, s2):
        n = sum(s1.values())
        assert n == sum(s2.values())
        return sum(
            max(0.0, (s1[k] - math.exp(privacy_parameter) * s2.get(k, 0)) / n)
            for k in s1.keys()
        )

    test_stat1 = dp_test_statistic(sample_hist1, sample_hist2)
    test_stat2 = dp_test_statistic(sample_hist2, sample_hist1)

    tolerance = 0.0025
    assert test_stat1 < tolerance
    assert test_stat2 < tolerance
