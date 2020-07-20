from assertpy import assert_that
from hypothesis import given
from hypothesis.strategies import lists
from hypothesis.strategies import integers
import pytest

import adaptive_group_testing


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
        subjects, intersecting_test([10]), 1
    )

    assert_that(actual).is_equal_to([10])


def test_reversed_find_10():
    subjects = list(reversed(range(100)))
    actual = adaptive_group_testing.generalized_binary_split(
        subjects, intersecting_test([10]), 1
    )

    assert_that(actual).is_equal_to([10])
