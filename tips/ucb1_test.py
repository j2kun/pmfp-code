from dataclasses import dataclass
import random

from assertpy import assert_that

from tips.ucb1 import ucb1


def test_find_best_action():
    random.seed(123)

    @dataclass(eq=True)
    class CoinFlipTestAction:
        p: float  # probability of heads

    def action_reward(action):
        return 1 if random.random() < action.p else 0

    actions = [
        CoinFlipTestAction(p=0.1),
        CoinFlipTestAction(p=0.2),
        CoinFlipTestAction(p=0.3),
        CoinFlipTestAction(p=0.4),
    ]

    best_action = actions[-1]
    generator = ucb1(actions, action_reward)

    for i in range(5000):
        next(generator)

    assert_that(next(generator)).is_equal_to(best_action)


def test_discriminate_between_two_close_actions():
    random.seed(123)

    @dataclass(eq=True)
    class CoinFlipTestAction:
        p: float  # probability of heads

    def action_reward(action):
        return 1 if random.random() < action.p else 0

    actions = [
        CoinFlipTestAction(p=0.49),
        CoinFlipTestAction(p=0.51),
    ]

    best_action = actions[-1]
    generator = ucb1(actions, action_reward)

    for i in range(10000):
        next(generator)

    assert_that(next(generator)).is_equal_to(best_action)
