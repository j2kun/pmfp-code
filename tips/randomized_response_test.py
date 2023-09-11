import math

from hypothesis import example, given
from hypothesis.strategies import integers

from tips.randomized_response import aggregate_responses, respond_privately


@given(integers(min_value=25, max_value=1000), integers(min_value=25, max_value=1000))
@example(200, 0)
@example(0, 200)
@example(200, 200)
@example(200, 50)
@example(50, 200)
def test_private_response(false_count, true_count):
    n = false_count + true_count
    true_mean = true_count / n
    true_variance = 3 / (4 * n)

    true_answers = [True] * true_count + [False] * false_count
    responses = [respond_privately(ans) for ans in true_answers]
    mean, claimed_variance = aggregate_responses(responses)

    deviation_bound_true = 3 * math.sqrt(true_variance)
    deviation_bound_claimed = 3 * math.sqrt(claimed_variance)

    # the estimate is within 2 standard deviations both using the claimed
    # variance and the true variance
    assert abs(mean - true_mean) < deviation_bound_true
    assert abs(mean - true_mean) < deviation_bound_claimed
