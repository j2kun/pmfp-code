from collections import Counter
from hypothesis import given
from hypothesis import settings
from hypothesis.strategies import floats
from hypothesis.strategies import integers
from hypothesis.strategies import lists
from hypothesis.strategies import text 
import math

from count_min_sketch import CountSketch


@settings(deadline=2000)
@given(
    lists(integers(), min_size=5),
    floats(min_value=1e-04, max_value=0.5, allow_nan=False, allow_infinity=False)
)
def test_error_guarantee_achieved(values, accuracy):
    sketch = CountSketch(accuracy=accuracy, confidence=1e-10)
    for value in values:
        sketch.increment(value)

    for (value, true_count) in Counter(values).items():
        est_count = sketch.count(value)
        assert true_count <= est_count
        assert est_count <= true_count + sketch.additive_error_guarantee()


@given(lists(text()))
def test_len(values):
    sketch = CountSketch(accuracy=0.01, confidence=0.01)
    for value in values:
        sketch.increment(value)
    assert len(sketch) == len(values)
