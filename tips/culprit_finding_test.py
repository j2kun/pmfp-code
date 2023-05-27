import random

from hypothesis import given
from hypothesis import settings
from hypothesis.strategies import composite
from hypothesis.strategies import floats
from hypothesis.strategies import integers
from hypothesis.strategies import lists

from tips.culprit_finding import find_culprits
from tips.culprit_finding import Change


def make_test_fn(culprit: Change, true_flake_rate: float):
    def test_fn(change: Change) -> bool:
        # passes if and only if it comes before the culprit
        true_result = change.id < culprit.id
        # only possibly incorrectly report False if the test passes
        if true_result and random.random() < true_flake_rate:
            return False
        return true_result

    return test_fn


def test_trivial():
    assert find_culprits(lambda _: True, [], (1.0,), 0.01) is None


def test_simple():
    random.seed(1)
    changes = [
        Change(id=1),
        Change(id=3),
        Change(id=7),
    ]
    prior = (0.25, 0.25, 0.25, 0.25)
    culprit = changes[1]
    true_flake_rate = 0.01
    test_fn = make_test_fn(culprit, true_flake_rate)
    actual = find_culprits(test_fn, changes, prior, true_flake_rate)
    assert culprit == actual


def test_no_culprit():
    random.seed(1)
    changes = [Change(id=i) for i in range(1000)]
    dist_len = 1001
    prior = [1.0 / dist_len] * dist_len
    true_flake_rate = 0.1

    def fail_only_for_flakes(change: Change) -> bool:
        del change  # unused
        return random.random() > true_flake_rate

    actual = find_culprits(fail_only_for_flakes, changes, prior, true_flake_rate)
    assert actual is None


@composite
def hidden_culprit(
    draw,
    changes=lists(
        integers(min_value=1, max_value=100), min_size=3, max_size=100, unique=True
    ),
):
    changes = [Change(id=i) for i in draw(changes)]
    culprit_index = draw(integers(min_value=0, max_value=len(changes) - 1))
    return (changes, changes[culprit_index])


@given(
    hidden_culprit(),
    floats(min_value=1e-04, max_value=0.1, allow_infinity=False, allow_nan=False),
    integers(min_value=0, max_value=1000),
)
@settings(print_blob=True)
def test_uniform_prior_and_true_flake_rate(changes_and_culprit, true_flake_rate, seed):
    random.seed(seed)
    changes, culprit = changes_and_culprit
    dist_len = 1 + len(changes)
    prior = tuple([1.0 / dist_len] * dist_len)

    test_fn = make_test_fn(culprit, true_flake_rate)
    actual = find_culprits(test_fn, changes, prior, true_flake_rate)
    assert culprit == actual


@given(
    hidden_culprit(),
    floats(min_value=0, max_value=0.1, allow_infinity=False, allow_nan=False),
    floats(min_value=-0.05, max_value=0.05, allow_infinity=False, allow_nan=False),
    integers(min_value=0, max_value=1000),
)
@settings(print_blob=True)
def test_uniform_prior_and_inaccurate_estimated_flakiness_rate(
    changes_and_culprit, true_flake_rate, est_deviation, seed
):
    random.seed(seed)
    changes, culprit = changes_and_culprit
    dist_len = 1 + len(changes)
    prior = tuple([1.0 / dist_len] * dist_len)

    # if the estimated flake rate is zero, then
    estimated_flake_rate = max(1e-04, true_flake_rate + est_deviation)
    test_fn = make_test_fn(culprit, true_flake_rate)
    actual = find_culprits(test_fn, changes, prior, estimated_flake_rate)
    assert culprit == actual


@composite
def hidden_culprit_and_prior(
    draw,
    changes=lists(
        integers(min_value=1, max_value=100), min_size=3, max_size=100, unique=True
    ),
    prior_deviations=floats(
        min_value=-0.5, max_value=0.5, allow_infinity=False, allow_nan=False
    ),
):
    changes = [Change(id=i) for i in draw(changes)]
    culprit_index = draw(integers(min_value=0, max_value=len(changes) - 1))
    dist_len = 1 + len(changes)

    prior = [1.0 / dist_len] * dist_len
    for i in range(dist_len):
        # increase or decrease by at most 50% of the uniform value
        prior[i] += draw(prior_deviations) * 1.0 / dist_len

    # renormalize
    total = sum(prior)
    prior = tuple([p / total for p in prior])

    return (changes, changes[culprit_index], prior)


@given(
    hidden_culprit_and_prior(),
    floats(min_value=0, max_value=0.1, allow_infinity=False, allow_nan=False),
    floats(min_value=-0.05, max_value=0.05, allow_infinity=False, allow_nan=False),
    integers(min_value=0, max_value=1000),
)
@settings(print_blob=True)
def test_arbitrary_prior_and_inaccurate_flake_rate(
    changes_culprit_prior, true_flake_rate, est_deviation, seed
):
    random.seed(seed)
    changes, culprit, prior = changes_culprit_prior

    # if the estimated flake rate is zero, then
    estimated_flake_rate = max(1e-04, true_flake_rate + est_deviation)
    test_fn = make_test_fn(culprit, true_flake_rate)
    actual = find_culprits(test_fn, changes, prior, estimated_flake_rate)
    assert culprit == actual
