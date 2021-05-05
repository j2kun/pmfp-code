from collections import Counter
from hypothesis import assume
from hypothesis import given
from hypothesis import settings
from hypothesis.strategies import floats
from hypothesis.strategies import integers
from hypothesis.strategies import lists
from hypothesis.strategies import text
import math
import random

from count_min_sketch import CountSketch
from count_min_sketch import heavy_hitters


@settings(deadline=2000)
@given(
    lists(integers(), min_size=5),
    floats(min_value=1e-04, max_value=0.5, allow_nan=False, allow_infinity=False)
)
def test_sketch_error_guarantee_achieved(values, accuracy):
    sketch = CountSketch(accuracy=accuracy, confidence=1e-10)
    for value in values:
        sketch.increment(value)

    for (value, true_count) in Counter(values).items():
        est_count = sketch.count(value)
        assert true_count <= est_count
        assert est_count <= true_count + sketch.additive_error_guarantee()


@given(lists(text()))
def test_sketch_len(values):
    sketch = CountSketch(accuracy=0.01, confidence=0.01)
    for value in values:
        sketch.increment(value)
    assert len(sketch) == len(values)


@given(integers(min_value=2, max_value=20))
def test_heavy_hitters_guarantee_achieved(k):
    n = 100
    elements = list(range(1, n+1))
    threshold = n // (2 * k)

    heavy_hitter = elements[0]
    almost_heavy = elements[1:3]

    data = [heavy_hitter] * ((n // k) + 1)
    data += almost_heavy * (threshold + 1)

    underweight = elements[3:]
    i = 0
    while len(data) < n:
        data.extend([underweight[i]] * min(threshold - 1, n - len(data)))
        i += 1

    data = data[:n]

    random.shuffle(data)
    output = heavy_hitters(data, k)

    assert heavy_hitter in output, str(Counter(data))
    for x in set(underweight):
        assert x not in output, str(Counter(data))

