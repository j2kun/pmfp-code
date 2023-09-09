import math

from assertpy import assert_that
from hypothesis import given
from hypothesis.strategies import integers
from hypothesis.strategies import lists

import tips.adaptive_group_testing as adaptive_group_testing


class TrackTestCounts:
    def __init__(self, f):
        self.f = f
        self.call_count = 0

    def __call__(self, *args, **kwargs):
        self.call_count += 1
        return self.f(*args, **kwargs)


def intersecting_test(S):
    # return a callable that returns true if any member is in S
    @TrackTestCounts
    def intersects_S(L):
        return bool(set(S) & set(L))

    return intersects_S


def test_find_10():
    subjects = list(range(100))
    actual = adaptive_group_testing.generalized_binary_split(
        subjects,
        intersecting_test([10]),
        1,
    )

    assert_that(actual).is_equal_to([10])


def test_reversed_find_10():
    subjects = list(reversed(range(100)))
    actual = adaptive_group_testing.generalized_binary_split(
        subjects,
        intersecting_test([10]),
        1,
    )

    assert_that(actual).is_equal_to([10])


def test_look_for_two_positive():
    subjects = list(range(25))
    test = intersecting_test([1, 0])

    actual = adaptive_group_testing.generalized_binary_split(subjects, test, 2)
    assert_that(actual).contains_only(1, 0)


@given(lists(integers(min_value=0, max_value=24), min_size=1, max_size=10, unique=True))
def test_finds_specific_subjects_with_exact_bound(positive_subjects):
    subjects = list(range(25))

    test = intersecting_test(positive_subjects)
    actual = adaptive_group_testing.generalized_binary_split(
        subjects,
        test,
        len(positive_subjects),
    )

    assert_that(actual).contains_only(*positive_subjects)


@given(
    lists(integers(min_value=0, max_value=999), min_size=1, max_size=20, unique=True),
)
def test_optimality(positive_subjects):
    subjects = list(range(1000))
    d = len(positive_subjects)
    n = len(subjects)

    test = intersecting_test(positive_subjects)
    _ = adaptive_group_testing.generalized_binary_split(
        subjects,
        test,
        len(positive_subjects),
    )

    # this is a weak assertion because the number of tests only approaches
    # d * log2(n/d) in expectation.
    expected_count = d * math.log2(n / d)
    assert_that(test.call_count / expected_count).is_less_than(1.5)
