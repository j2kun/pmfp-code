import itertools

import numpy as np
import pytest

from tips.decision_stump import DecisionStump, compute_error, train_decision_stump


def random_examples(dim=10, n=2000):
    return np.random.uniform(size=(n, dim))


@pytest.mark.parametrize("chosen_feature", [1, 3, 4, 7])
def test_stump_finds_chosen_feature(chosen_feature):
    dim = 10
    data = [(x, 1 if x[chosen_feature] > 0.5 else -1) for x in random_examples(dim=dim)]
    hypothesis = train_decision_stump(data.__iter__())

    alternatives = [
        DecisionStump(
            gt_label=label,
            lt_label=-label,
            threshold=0.5,
            feature_index=index,
        )
        for (index, label) in itertools.product(range(dim), [-1, 1])
        if index != chosen_feature
    ]

    alt_errors = [compute_error(data, alt.classify) for alt in alternatives]
    actual_error = compute_error(data, hypothesis)

    assert actual_error < min(alt_errors)
