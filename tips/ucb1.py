from typing import Callable
from typing import Generator
from typing import List
from typing import TypeVar
import math


# just for type clarity
Action = TypeVar('Action')
RewardFn = Callable[[Action], float]


def ucb1(actions: List[Action],
         reward: RewardFn) -> Generator[Action, None, None]:
    '''The Upper Confidence Bound 1 (UCB1) algorithm.

    This function is an infinite generator that chooses an action to take
    in each step, the rewards for which are produced by the `reward` callable.
    UCB1 balances exploration (trying different actions) and exploitation
    (playing actions with large rewards).

    UCB1 guarantees that if each action corresponds to a fixed(but arbitrary)
    distribution with values in [0, 1], and the rewards are drawn
    independently from these distributions in each round, then the expected
    regret of UCB1(compared to the best single action in hindsight) has the
    order of magnitude sqrt(K * T * log(T)), where K is the number of actions
    and T is the number of rounds.

    Arguments:
      - actions: a list of actions that can be taken by the algorithm
      - reward: a callable accepting as input a single action and producing
          as output a float representing the reward or cost of the action.

    Returns:
      A generator yielding an infinite stream of actions.
    '''
    num_actions: int = len(actions)
    payoff_sums: List[float] = [0] * num_actions

    # Play each action once to initialize empirical sums.
    for i, action in enumerate(actions):
        payoff_sums[i] = reward(action)
        yield action

    num_plays: List[int] = [1] * num_actions
    t: int = num_actions

    def upperBound(step: int, num_plays: int) -> float:
        '''Return the margin of the confidence bound from its mean.

        This method does not need to know the expected value of the action
        being attempted. The confidence bound depends only on the number of
        total actions attempted and the number of times one particular action
        has been tried.
        '''
        return math.sqrt(2 * math.log(step) / num_plays)

    while True:
        upper_confidence_bounds: List[float] = [
            payoff_sums[i] / num_plays[i] + upperBound(t, num_plays[i])
            for i in range(num_actions)
        ]
        chosen_action_index: int = max(
            range(num_actions),
            key=lambda i: upper_confidence_bounds[i]
        )
        chosen_action: Action = actions[chosen_action_index]

        num_plays[chosen_action_index] += 1
        payoff_sums[chosen_action_index] += reward(chosen_action)
        t = t + 1
        yield chosen_action
