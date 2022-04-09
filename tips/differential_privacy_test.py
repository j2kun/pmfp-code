import math
from collections import Counter
from collections import defaultdict

from hypothesis import strategies as st
from hypothesis import given
from hypothesis import example
import numpy as np

from differential_privacy import LaplaceGenerator
from differential_privacy import privatize_histogram


class NumpyLaplaceGenerator(LaplaceGenerator):
    def sample(self, scale) -> float:
        return np.random.default_rng().laplace(0, scale, 1)[0]


@st.composite
def neighboring_histograms(draw, elements, min_size=1, max_size=10):
    hist = draw(st.lists(elements, min_size=min_size, max_size=max_size))
    hist1 = tuple(hist)

    index = draw(st.integers(min_value=0, max_value=len(hist) - 1))
    change = draw(st.sampled_from([-1, 1]))
    hist[index] += change
    hist2 = tuple(hist)

    return (hist1, hist2)


# @given(neighboring_histograms(st.integers(min_value=0, max_value=100)),
#        st.floats(min_value=0.001, max_value=0.5, allow_infinity=False,
#                  allow_nan=False),
#         )
# @example(((1, 2, 1, 2), (1, 2, 2, 2)), 0.5)
def test_privatize_histogram():  # neighboring_hists, privacy_parameter):
    neighboring_hists, privacy_parameter = ((1,), (2,)), math.log(3)

    rng = NumpyLaplaceGenerator()
    hist1, hist2 = neighboring_hists
    sample_size = 200000

    def sample(hist):
        return Counter(
            [tuple(privatize_histogram(hist, privacy_parameter, rng))
             for _ in range(sample_size)])

    sample_hist1 = sample(hist1)
    sample_hist2 = sample(hist2)

    def dp_test_statistic(s1, s2):
        return sum(
            max(0.0, (s1[k] - math.exp(privacy_parameter) * s2.get(k, 0)) / len(s1))
            for k in s1.keys()
        )

    test_stat1 = dp_test_statistic(sample_hist1, sample_hist2)
    test_stat2 = dp_test_statistic(sample_hist2, sample_hist1)

    tolerance = 0.0025
    assert test_stat1 < tolerance 
    assert test_stat2 < tolerance
