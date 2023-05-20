"""An implementation of a simple decision stump learner for use in tips/boosting.py."""

from collections import Counter
from dataclasses import dataclass
from typing import Callable
from typing import Iterator

LabeledExample = tuple[list[float], int]
Dataset = list[LabeledExample]
Hypothesis = Callable[[list[float]], int]
ErrorFn = Callable[[Dataset, Hypothesis], float]
DrawIter = Iterator[LabeledExample]


@dataclass
class DecisionStump:
    gt_label: int = 1
    lt_label: int = -1
    threshold: float = 0.0
    feature_index: int = 0

    def classify(self, point):
        return (
            self.gt_label
            if point[self.feature_index] >= self.threshold
            else self.lt_label
        )


def most_common_label(data: Dataset) -> int:
    """Compute the label that occurs most commonly in the set of labeled examples."""
    return Counter([label for (_, label) in data]).most_common(1)[0][0]


def compute_error(data: Dataset, h: Hypothesis) -> float:
    """Compute the min of the decision stump's error and its negation's error."""
    h_pos = [(x, y) for (x, y) in data if h(x) == 1]
    h_neg = [(x, y) for (x, y) in data if h(x) == -1]

    error = sum(y == -1 for (x, y) in h_pos) + sum(y == 1 for (x, y) in h_neg)
    negated_error = sum(y == 1 for (x, y) in h_pos) + sum(y == -1 for (x, y) in h_neg)
    return min(error, negated_error) / len(data)


@dataclass
class ThresholdResult:
    feature_index: int
    threshold: float
    error: float


def best_threshold_for_feature(
    data: Dataset, index: int, error_fn: ErrorFn
) -> ThresholdResult:
    """Compute best threshold for a given feature."""
    thresholds = [point[index] for (point, label) in data]

    errors = {
        t: error_fn(data, DecisionStump(feature_index=index, threshold=t).classify)
        for t in thresholds
    }
    best_threshold = min(errors, key=lambda x: errors[x])
    return ThresholdResult(
        feature_index=index, threshold=best_threshold, error=errors[best_threshold]
    )


def train_decision_stump(draw_example: DrawIter, debug: bool = True):
    data = [next(draw_example) for _ in range(500)]
    num_features = len(data[0][0])

    best_thresholds = [
        best_threshold_for_feature(data, i, compute_error) for i in range(num_features)
    ]
    best_threshold_result = min(best_thresholds, key=lambda t: t.error)

    thresh = best_threshold_result.threshold
    feature = best_threshold_result.feature_index
    gt_label = most_common_label([x for x in data if x[0][feature] >= thresh])
    lt_label = most_common_label([x for x in data if x[0][feature] < thresh])

    stump = DecisionStump(
        feature_index=feature, threshold=thresh, gt_label=gt_label, lt_label=lt_label
    )

    if debug:
        print(f"Feature: {feature}, threshold: {thresh}, {stump.gt_label}\n")

    return stump
