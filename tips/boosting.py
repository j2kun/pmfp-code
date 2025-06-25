"""An implementation of AdaBoost for binary classification."""

import math
import random
from typing import Callable, Iterator

LabeledExample = tuple[list[float], int]
Dataset = list[LabeledExample]
Hypothesis = Callable[[list[float]], int]
ErrorFn = Callable[[Dataset, Hypothesis], float]
DrawIter = Iterator[LabeledExample]
Learner = Callable[[DrawIter], Hypothesis]


def sign(x: float) -> int:
    return 1 if x >= 0 else -1


def normalize(weights: list[float]) -> tuple[float, ...]:
    """Normalize a list of floats into a distribution."""
    norm = sum(weights)
    return tuple(m / norm for m in weights)


def compute_error(h: Hypothesis, examples: Dataset) -> float:
    """Compute the absolute error of a hypothesis on a dataset."""
    prediction_results = [h(x) * y for (x, y) in examples]  # +1 if correct, else -1
    return len([x for x in prediction_results if x == -1]) / len(examples)


class DrawExample:
    def __init__(self, distr, examples):
        self.distr = distr
        self.examples = examples

    def __iter__(self):
        return self  # pragma: no cover

    def __next__(self):
        weights = self.distr
        index = random.choices(list(range(len(weights))), weights=weights)[0]
        return self.examples[index]


def boost(examples: Dataset, weak_learner: Learner, rounds: int) -> Hypothesis:
    """Boost the accuracy of a weak learner."""
    distr = normalize([1.0] * len(examples))
    hypotheses: list[Hypothesis] = []
    alphas: list[float] = []

    for t in range(rounds):
        hypothesis = weak_learner(DrawExample(distr, examples))
        hypotheses.append(hypothesis)
        prediction_results = [
            hypothesis(x) * y for (x, y) in examples
        ]  # +1 if correct, else -1
        weighted_error = sum(w for (z, w) in zip(prediction_results, distr) if z < 0)

        alpha = 0.5 * math.log((1 - weighted_error) / (0.0001 + weighted_error))
        alphas.append(alpha)
        distr = normalize(
            [d * math.exp(-alpha * h) for (d, h) in zip(distr, prediction_results)],
        )
        print("Round %d, error %.3f" % (t, weighted_error))

    def final_hypothesis(x):
        return sign(sum(a * h(x) for (a, h) in zip(alphas, hypotheses)))

    print(
        "Final hypothesis training error %.3f"
        % (compute_error(final_hypothesis, examples)),
    )

    return final_hypothesis
