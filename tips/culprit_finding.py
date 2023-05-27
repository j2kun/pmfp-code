"""An implementation of flake-aware culprit finding."""

from dataclasses import dataclass
from typing import Callable
from typing import Optional


@dataclass
class Change:
    id: int


TestFn = Callable[[Change], bool]
"""
A TestFn represents the result of running a test at a specific change. In this
demo, changes are linearly ordered by their id, and the TestFn may return an
incorrect answer in one direction only. I.e., if the result is True, then
change did not introduce a bug. If the result is False, then the test may fail
because it is flaky, or because the change introduced a bug.
"""

Distribution = tuple[float, ...]
"""
A Distribution has length equal to one plus the number of changes considered
suspect. Each entry represents the probability that the change introduced the
culprit. The last entry represents the probability that there is no culprit.
"""


def find_culprits(
    test_fn: TestFn,
    suspects: list[Change],
    prior: Distribution,
    flakiness: float,
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
        - flakiness: An estimate of the flakiness rate of test_fn

    Returns:
      A single change that is the culprit, or None if no culprit is found.
    """
    assert len(suspects) + 1 == len(prior)
    return None
