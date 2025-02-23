import numpy as np

import data.adult as adult
from tips.boosting import boost, compute_error
from tips.decision_stump import train_decision_stump


def test_learn_line():
    examples = np.random.uniform(size=(500, 2))

    def true_label(x):
        return 1 if 2 * x[0] > 3 * x[1] + 0.7 else -1

    dataset = [(x, true_label(x)) for x in examples]

    h = boost(dataset, train_decision_stump, 25)

    assert compute_error(h, dataset) < 0.05


def test_adult_dataset():
    test, train = adult.load()
    h = boost(
        train,
        lambda draw: train_decision_stump(draw, sample_size=100),
        rounds=15,
    )
    assert compute_error(h, test) < 0.165
