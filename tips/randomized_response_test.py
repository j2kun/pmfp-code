import math

from hypothesis import given
from hypothesis import example
from hypothesis.strategies import booleans
from hypothesis.strategies import lists

from randomized_response import respond_privately
from randomized_response import aggregate_responses


@given(lists(booleans(), min_size=50, max_size=1000))
@example([False] * 200)
@example([True] * 200)
@example([False] * 200 + [True] * 200)
@example([False] * 200 + [True] * 50)
@example([False] * 50 + [True] * 200)
def test_private_response(true_answers):
    n = len(true_answers)
    true_mean = sum(true_answers) / n
    true_variance = true_mean * (1 - true_mean) / n + 1 / n

    # allow 2 * true_stddev estimator error to simulate a confidence interval
    allowed_error = 2 * math.sqrt(true_variance)

    responses = [respond_privately(ans) for ans in true_answers]
    mean, variance = aggregate_responses(responses)

    # the estimate is within 2 standard deviations
    assert abs(mean - true_mean) < allowed_error

    # the estimate of standard deviation is within 3%
    assert abs(math.sqrt(variance) - math.sqrt(true_variance)) < 0.03
