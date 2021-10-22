import random
import pytest

from loglog import LogLog
from loglog import cardinality
from loglog import trailing_zeros


estimator_names = LogLog(log_bucket_count=1).estimator_names


def random_data(n=100000):
    return [random.random() for i in range(n)]


def test_trailing_zeros_default():
    assert trailing_zeros(0, 7) == 7


@pytest.mark.parametrize("estimator_name", estimator_names)
def test_small_error_in_good_cardinality_range(estimator_name):
    random.seed(1)

    n = 100000
    for i in range(5):
      result = cardinality(random_data(n=n), estimator=estimator_name) / n

      assert abs(1 - result) < 0.04


def test_improved_estimator_works_on_all_cardinalities():
    random.seed(1)

    for n in [1, 10, 100, 1000, 10000, 100000, 1000000]:
      result = cardinality(random_data(n=n), estimator='improved') / n
      assert abs(1 - result) < 0.04


@pytest.mark.parametrize("estimator_name", [x for x in estimator_names if x != 'improved'])
def test_normal_estimators_are_poor_on_small_cardinalities(estimator_name):
    random.seed(1)

    for n in [1, 10, 100, 1000, 10000]:
      result = cardinality(random_data(n=n), estimator=estimator_name) / n
      assert abs(1 - result) > 0.04
