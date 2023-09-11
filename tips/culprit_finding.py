"""An implementation of flake-aware culprit finding."""

from dataclasses import dataclass, replace
from typing import Callable, Optional


@dataclass(eq=True, frozen=True)
class Change:
    id: int


# A TestFn represents the result of running a test at a specific change. In this
# demo, changes are linearly ordered by their id, and the TestFn may return an
# incorrect answer in one direction only. I.e., if the result is True, then
# change did not introduce a bug. If the result is False, then the test may fail
# because it is flaky, or because the change introduced a bug.
TestFn = Callable[[Change], bool]


@dataclass(frozen=True)
class Distribution:
    """A Distribution has length equal to one plus the number of changes considered
    suspect.

    Each entry represents the probability that the change introduced the culprit. The
    last entry represents the probability that there is no culprit.
    """

    probs: dict[Change, float]
    flake_rate: float

    def __iter__(self):
        return iter(self.probs)

    def __str__(self):
        return ", ".join(
            f"{change.id}={prob:.3f}" for (change, prob) in self.probs.items()
        )

    def update_pass(self, tested_change: Change) -> "Distribution":
        """Bayesian update after observing a passed test."""

        def conditional(change: Change) -> float:
            # Recall: changes are linearly ordered by their id. This function
            # computes the conditional, i.e.,
            #
            #   Prob[test passed at `tested_change` | culprit is `change`]
            #
            if change.id <= tested_change.id:
                # A passing test implies this change and no prior change
                # can be the culprit.
                return 0
            else:
                # otherwise, a future re-run may fail due to being a flake
                # Prob[observed Pass at `tested_change` | culprit is later]
                #     = 1 - flake_rate
                return 1 - self.flake_rate

        # The normalization factor: the probability that this outcome is
        # observed given the prior distribution on culprit probabilities.
        # Uses the so-called "law of total probability" to expand the
        # P[observed Pass at `tested_change`] into a sum over all possible
        # choices of culprits, where each term only relies on the conditional
        # and prior probabilities.
        #
        # In our case this could be optimized by filtering over the range of
        # changes for which conditinal(change) returns a non-zero value, but
        # for explanatory purposes I'll leave it in this more verbose form.
        bayes_rule_denominator = sum(
            conditional(change) * prob for (change, prob) in self.probs.items()
        )

        # Now apply Bayes rule directly
        new_probs = {
            change: conditional(change) * self.probs[change] / bayes_rule_denominator
            for (change, prob) in self.probs.items()
        }

        return replace(self, probs=new_probs)

    def update_fail(self, tested_change: Change) -> "Distribution":
        """Bayesian update after observing a failed test."""

        # update_pass has a detailed breakdown, and here I just put the comments
        # for what's different from update_pass.
        def conditional(change: Change) -> float:
            if change.id <= tested_change.id:
                # If the culprit comes earlier, then all later changes will show
                # a failing test.
                return 1
            else:
                # If the culprit comes after, we might just be observing a flake.
                return self.flake_rate

        bayes_rule_denominator = sum(
            conditional(change) * prob for (change, prob) in self.probs.items()
        )
        new_probs = {
            change: conditional(change) * self.probs[change] / bayes_rule_denominator
            for (change, prob) in self.probs.items()
        }
        return replace(self, probs=new_probs)


def next_change_to_test(
    distribution: Distribution,
    threshold=0.5,
    tested_changes: Optional[set[Change]] = None,
) -> Optional[Change]:
    """Determine the next change to test.

    Do this by converting the (linearly ordered) distribution of culprit probabilities
    into a cumulative distribution, and then finding the first change for which the
    cumulative probability of a culprit preceding that change is 0.5.

    This function makes optional an optimization from the Henderson paper, which is to
    test the run preceding the one to try next, the idea being: if you think the next
    change to test is going to fail, you'll get more information by first trying the
    preceding test (which you think will pass) and then trying the test that you think
    will fail. This takes advantage of the information asymmetry of passing and failing
    tests. If tested_changes is passed and nonempty, then it will apply this
    optimization.
    """
    epsilon = 1e-08
    # to avoid floating point roundoff errors
    assert threshold < 1 - epsilon

    selected_change = None
    sorted_dist = sorted(distribution.probs.items(), key=lambda kv: kv[0].id)
    cumulative_prob = 0.0
    for i, (change, prob) in enumerate(sorted_dist):
        cumulative_prob += prob
        if cumulative_prob >= threshold - epsilon:
            selected_change = change
            break

    # If this fails, the probabilities don't sum to 1.
    assert selected_change is not None

    # The Henderson paper optimization to use the prior test if it hasn't
    # already been run.
    if i - 1 >= 0:
        previous_change, previous_prob = sorted_dist[i - 1]
        if (
            previous_prob > 0
            and tested_changes
            and previous_change not in tested_changes
        ):
            return previous_change

    # The sentinel was chosen, i.e., more than 50% chance that there is no
    # culprit.
    if selected_change == sorted_dist[-1][0]:
        return None

    return selected_change


def find_culprits(
    test_fn: TestFn,
    suspects: list[Change],
    prior: tuple[float, ...],
    flakiness: float,
    exit_threshold: float = 1 - 1e-10,
) -> Optional[Change]:
    """Find the change that causes a test to fail among a list of suspects.

    The suspects are linearly ordered by their id. The test has a known prior
    flake rate, and the test is assumed to pass on a change if and only if the
    change (and all changes before it) are not the culprit. I.e., tests are
    assumed to not "flakily pass."

    Args:
        - test_fn: A function that runs a test on a suspected culprit, and
          returns True if the test passes.
        - suspects: A list of suspect changes to test.
        - prior: The prior distribution that suspects are the culprit.
            The values are assumed to be in the same order as the suspects,
            i.e., prior[i] is the prior probability that suspects[i] is the
            culprit.
        - flakiness: An estimate of the flakiness rate of test_fn
        - exit_threshold: Stop once the most likely culprit has a probability
            exceeding this threshold.

    Returns:
      A single change that is most likely to be the culprit, or None if the
      most likely outcome is that none of the suspects are the culprit.
    """
    assert len(suspects) + 1 == len(prior)
    assert abs(sum(prior) - 1) < 1e-06

    # If flakiness estimate is zero (and the true flakiness rate is nonzero),
    # then we will treat failing tests as 100% accurate, which omits the
    # reality that the flakiness can only be estimated. One could imagine
    # extending this to make the flakiness estimate also have uncertainty,
    # but for simplicity we don't. Doing so would lead us more towards the
    # graphical Bayesian networks in TrueSkill.
    assert flakiness > 0

    if not suspects:
        return None

    tested_changes: set[Change] = set()
    sentinel = Change(id=1 + max(c.id for c in suspects))
    dist = Distribution(
        probs=dict(zip(suspects + [sentinel], prior)),
        flake_rate=flakiness,
    )

    most_likely_culprit = max(dist, key=lambda c: dist.probs[c])
    print("")
    while dist.probs[most_likely_culprit] < exit_threshold:
        print(f"most_likely_culprit={most_likely_culprit}, dist={dist}")
        next_change = next_change_to_test(dist, tested_changes=tested_changes)
        if not next_change:
            # next_change_to_test will return None if "no culprit" has more
            # than 50% probability. In this case we still want to continue
            # testing, so we use the last change by default. If the last
            # change passes, then no_culprit will exceed the exit_threshold,
            # otherwise next_change_to_test will eventually stop returning
            # the sentinel.
            if dist.probs[sentinel] < exit_threshold:
                next_change = max(
                    dist.probs.keys(),
                    key=lambda c: 0 if c == sentinel else dist.probs[c],
                )
            else:
                # If it exceeded the exit_threshold, the loop would have
                # exited.
                raise ValueError("unreachable")  # pragma: no cover

        if test_fn(next_change):
            print(f"Test at {next_change.id} passed.")
            dist = dist.update_pass(next_change)
        else:
            print(f"Test at {next_change.id} failed.")
            dist = dist.update_fail(next_change)

        tested_changes.add(next_change)
        most_likely_culprit = max(dist, key=lambda c: dist.probs[c])

    print(f"most_likely_culprit={most_likely_culprit}, dist={dist}")
    return None if most_likely_culprit == sentinel else most_likely_culprit
