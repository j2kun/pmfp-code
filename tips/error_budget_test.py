from datetime import timedelta
from hypothesis import assume
from hypothesis import example
from hypothesis import given
from hypothesis.strategies import floats
from hypothesis.strategies import integers

from error_budget import error_budget_remaining
from error_budget import SloMetric


def test_no_errors_no_requests():
    requests = [100 for x in range(100)]
    errors = [0 for x in requests]
    expected = SloMetric(violated=False, budget_growth_rate=0.0)
    assert error_budget_remaining(requests, errors, 0.5, window_minutes=10) == expected


@given(
    floats(min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False),
    integers(min_value=1, max_value=999))
def test_error_rate_exactly_budget(budget, measurement_index):
    requests = [1000 + x for x in range(1000)]
    errors = [int(x * budget) for x in requests]
    expected = SloMetric(violated=True)
    assert error_budget_remaining(
        requests[:measurement_index],
        errors[:measurement_index],
        budget,
        window_minutes=10
    ) == expected


@given(
    floats(min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False),
    integers(min_value=1, max_value=999))
def test_error_rate_just_below_budget(budget, measurement_index):
    requests = [1000 + x for x in range(1000)]
    errors = [int(x * budget) - 1 for x in requests]
    actual = error_budget_remaining(
        requests[:measurement_index],
        errors[:measurement_index],
        budget,
        window_minutes=10
    )

    assert actual.violated == False
    assert actual.budget_growth_rate == 0


@given(
    floats(min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False),
    floats(min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False),
    integers(min_value=15, max_value=999),
)
@example(budget=0.026, error_rate=0.01, measurement_index=15)
def test_negative_budget_growth_rate(budget, error_rate, measurement_index):
    assume(budget > error_rate + 0.01)
    requests = [1000 + 100 * x for x in range(1000)]
    errors = [int(x * error_rate) for x in requests]
    actual = error_budget_remaining(
        requests[:measurement_index],
        errors[:measurement_index],
        budget,
        window_minutes=10
    )

    assert actual.violated == False
    assert actual.budget_growth_rate > 0


def test_positive_budget_growth_rate():
    requests = [1000 for _ in range(1000)]
    errors = list(range(1000))
    actual = error_budget_remaining(
        requests[:21],
        errors[:21],
        0.8,
        window_minutes=10
    )

    assert actual.violated == False
    assert actual.budget_growth_rate == -1

    # 780 minutes because our budget is 800 errors,
    # and we get one more error each time step.
    # measuring at index 20 means 780 remaining minutes
    assert actual.time_until_exhausted == timedelta(minutes=780)
