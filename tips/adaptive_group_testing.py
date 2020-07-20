from math import floor
from math import log2
from typing import Callable
from typing import List
from typing import Set
from typing import TypeVar

TestSubject = TypeVar('TestSubject')
Test = Callable[Set[TestSubject], bool]


def generalized_binary_split(
    test_subjects: List[TestSubject], test: Test, defective_count_bound: int
) -> List[TestSubject]:
    '''Find up to defective_count_bound subjects that test positively.

    This algorithm adaptively minimizes the number of tests performed. The
    input test must have the property that, when applied to a group of test
    subjects, it detects a positive result on the group if and only if some
    individual in the group tests positively.

    Arguments:
      - test_subjects: a list of test subjects remaining to test
      - test: the test to evaluate (groups of) subjects on
      - defective_count_bound: the maximum number of defects in test_subjects

    Returns:
      The sublist of test subjects that test positively
    '''
    if not test_subjects:
        return set()

    test_group, remainder = next_group_to_test(
        test_subjects, defective_count_bound
    )
    if test(test_group):
        positive, unknown = binary_search(test_group, test)
        return [positive] + generalized_binary_split(
            remainder + unknown, test, defective_count_bound - 1
        )
    else:
        return generalized_binary_split(remainder, test, defective_count_bound)


def next_group_to_test(
    test_subjects: List[TestSubject], defective_count_bound: int
) -> Tuple[List[TestSubject], List[TestSubject]]:
    '''Return the next group to test according to the Genralized Binary Splitting Algorithm.

    Arguments:
      - test_subjects: a list of test subjects remaining to test
      - defective_count_bound: the maximum number of defects in test_subjects

    Returns:
      A pair (S, X), where S is sublist of test_subjects to test,
      and X is test_subjects - S. The list S is always a prefix of test_subjects
    '''
    n = len(test_subjects)
    d = defective_count_bound
    if n <= 2 * d - 2:
        return (set(test_subjects[0]), test_subjects[1:])

    test_size = 2**(floor(log2((n - d + 1) / d)))
    return (test_subjects[:test_size], test_subjects[test_size:])
