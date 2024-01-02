"""An implementation of a patrol scheduling algorithm.

Similar to what is used in the PAWS project, see https://github.com/lily-x/mirror and
https://arxiv.org/abs/2106.08413
References to "the paper" and equation markers are in reference to that paper,
v1 submitted 2021-06-15.
"""

import random
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np
from scipy.ndimage import generic_filter
from scipy.stats import logistic

from util import ddpg


@dataclass
class PatrolProblem:
    """A patrol problem is defined by a grid representing a discretized map, and values
    that specify the wildlife activity at each grid cell, along with parameters related
    to the mathematical model of the problem."""

    # A 2D array of floats representing wildlife activity in the park.
    wildlife: np.ndarray

    # The total budget available to execute patrols.
    total_budget: float

    # learned poacher & environment model parameters

    # A parameter describing the natural growth rate of wildlife, constrained
    # to be greater than 1. Referred to as psi in the paper, see equation (6).
    wildlife_growth_ratio: float

    # A parameter describing the strength of poachers on reducing wildlife,
    # constrained to be greater than 0. Referred to as alpha in the paper, see
    # equation (6).
    poacher_strength: float

    # A parameter describing the effect of patrolling on reducing future
    # poaching. Less than zero if patrolling deters poaching. Referred to as
    # beta in the paper, see equations (3, 5).
    return_on_effort: float

    # A parameter describing the effect of patrolling on poaching in
    # neighboring cells. Greater than zero if patrolling incentivizes poachers
    # to poach elsewhere. Referred to as eta in the paper, see equation (5).
    displacement_effect: float


@dataclass
class PoacherPolicy:
    # A 2D array of floats representing how attractive a grid cell is for
    # poachers to attack. Referred to as z_i, and the entire set as Z. See
    # equation (5).
    #
    # Zero attractiveness corresponds to the middle range: equally likely
    # as not to place a snare there.
    #
    # Negative attractiveness crresponds to disincentive to place a snare
    # there. The values are passed through the logistic function.
    attractiveness: np.ndarray


@dataclass
class DefenderPolicy:
    # A 2D array of values in [0,1] representing how much effort to spend
    # patrolling a given grid cell. Referred to as a_i in the paper, this
    # class represents the policy for a single time step.
    patrol_effort: np.ndarray


def build_attack_model(
    poacher_policy: PoacherPolicy,
    patrol_problem: PatrolProblem,
    defender_policy: DefenderPolicy,
) -> np.ndarray:
    """Computes the probability of poaching activity in each cell."""
    footprint = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    # computes the sum of values of neighbors of each cell, using zero for
    # values that go outside the boundary of the array.
    neighbor_effort = generic_filter(
        defender_policy.patrol_effort,
        np.sum,
        footprint=footprint,
        mode="constant",
        cval=0,
    )
    logistic_input = (
        poacher_policy.attractiveness
        + patrol_problem.return_on_effort * defender_policy.patrol_effort
        + patrol_problem.displacement_effect * neighbor_effort
    )
    return logistic.cdf(logistic_input)


def draw_poacher_activity(attack_model: np.ndarray) -> np.ndarray:
    return np.random.binomial(1, attack_model)


def update_wildlife(
    wildlife: np.ndarray,
    poacher_activity: np.ndarray,
    defender_policy: DefenderPolicy,
    patrol_problem: PatrolProblem,
) -> np.ndarray:
    """Returns the new wildlife population in each cell."""
    # Note the input is the _current_ patrol effort vs the _past_ poacher
    # activity. Snares were set in the past, but might be detected by current
    # patrol.
    manmade_change = -(
        patrol_problem.poacher_strength
        * poacher_activity
        * (1 - defender_policy.patrol_effort)
    )
    natural_growth = wildlife**patrol_problem.wildlife_growth_ratio
    return np.maximum(0, natural_growth + manmade_change)


# A callback whose args are:
# - step number
# - current wildlife
# - probability of poacher attack
# - current defender policy
# - intermediate reward representing the expected wildlife at the next step
SimulationStepCallback = Callable[
    [int, np.ndarray, np.ndarray, DefenderPolicy, float],
    None,
]


def simulate_game(
    patrol_problem: PatrolProblem,
    draw_poacher_strategy: Callable[[], PoacherPolicy],
    draw_defender_strategy: Callable[[], DefenderPolicy],
    per_step_callable: Optional[SimulationStepCallback] = None,
    planning_horizon: int = 5,
) -> np.ndarray:
    """Simulate some number of rounds of the game.

    Args:
        patrol_problem: the problem to simulate
        draw_poacher_strategy: a function that returns a PoacherPolicy
        draw_defender_strategy: a function that returns a DefenderPolicy
        per_step_callable: a function that is called at each step of the game
        planning_horizon: the number of steps to simulate

    Returns:
        The wildlife at the end of the simulation
    """
    wildlife = patrol_problem.wildlife
    defender_policy = draw_defender_strategy()

    for step in range(planning_horizon):
        # simulate one round of the wildlife evolution
        poacher_policy = draw_poacher_strategy()
        p_attack = build_attack_model(
            poacher_policy=poacher_policy,
            patrol_problem=patrol_problem,
            # the last step patrol effort, which is what the poachers observe
            defender_policy=defender_policy,
        )
        poacher_activity = draw_poacher_activity(p_attack)

        defender_policy = draw_defender_strategy()
        assert (
            np.sum(defender_policy.patrol_effort) <= patrol_problem.total_budget + 1e-05
        ), (
            f"Total patrol effort {np.sum(defender_policy.patrol_effort)} exceeds "
            f"budget {patrol_problem.total_budget}"
        )
        next_wildlife = update_wildlife(
            wildlife,
            poacher_activity,
            defender_policy,
            patrol_problem,
        )

        # use the expected wildlife at this step as an intermediate reward
        reward = np.sum(
            update_wildlife(wildlife, p_attack, defender_policy, patrol_problem),
        )
        if per_step_callable:
            per_step_callable(step, wildlife, p_attack, defender_policy, reward)
        wildlife = next_wildlife

    return wildlife


class GameState:
    def __init__(self, patrol_problem: PatrolProblem):
        self.patrol_problem = patrol_problem
        self.shape = patrol_problem.wildlife.shape
        self.actions_dim = patrol_problem.wildlife.size

        initial_wildlife = patrol_problem.wildlife.flatten()
        initial_effort = np.zeros(shape=patrol_problem.wildlife.shape).flatten()
        initial_state = np.concatenate([initial_wildlife, initial_effort, [0]])
        self.state = initial_state

    def decompose_state(self):
        wildlife = self.state[: self.actions_dim].reshape(self.shape)
        effort = self.state[self.actions_dim : 2 * self.actions_dim].reshape(self.shape)
        step = self.state[-1]
        return wildlife, effort, step

    def update_state(self, wildlife, effort, step):
        old_state = self.state
        self.state = np.concatenate(
            [wildlife.flatten(), effort.flatten(), [step + 1]],
        )
        return (old_state, self.state)

    def draw_defender_policy(self, learner):
        sampled_defender_policy = learner.select_action(self.state).reshape(self.shape)
        # The sampled action is a softmax probability over the cells. We
        # interpret that as a budget allocation, but the model forces each
        # cell to have at most effort=1, so we clip anything exceeding 1.
        # This will impicitly teach the DDPG not to allocate any cell a
        # value more than budget / grid_size.
        next_effort = sampled_defender_policy * self.patrol_problem.total_budget
        next_effort[np.where(next_effort > 1)] = 1
        return DefenderPolicy(next_effort)


def defender_best_response(
    patrol_problem: PatrolProblem,
    poacher_strategies: np.ndarray,
    poacher_distribution: Sequence[float],
    training_rounds: int = 100,
    planning_horizon: int = 5,
):
    """Returns the defender's best response pure strategy for fixed attacker
    parameters."""
    # an action is a choice of patrol effort for each cell.
    actions_dim = patrol_problem.wildlife.size
    # a state is a concatenation of (wildlife, patrol effort, timestep index)
    num_states = 2 * actions_dim + 1
    learner = ddpg.DDPG(actions_dim=actions_dim, states_dim=num_states)

    for t in range(training_rounds):
        game_state = GameState(patrol_problem)

        def draw_poacher_policy() -> PoacherPolicy:
            sampled_poacher_policy = random.choices(
                poacher_strategies,
                weights=poacher_distribution,
            )[0]
            return PoacherPolicy(sampled_poacher_policy)

        def per_step_callable(step, wildlife, p_attack, defender_policy, reward):
            old_state, next_state = game_state.update_state(
                wildlife,
                defender_policy.patrol_effort,
                step,
            )
            learner.remember(
                old_state,
                defender_policy.patrol_effort.flatten(),
                np.array([reward]),
                next_state,
                step == planning_horizon - 1,
            )
            learner.update()

        simulate_game(
            patrol_problem,
            draw_poacher_policy,
            lambda: game_state.draw_defender_policy(learner),
            per_step_callable,
        )

    return learner


def nash_equilibrium(defender_strategies, attacker_strategies):
    pass


def schedule_patrols(patrol_problem: PatrolProblem, num_epochs: int = 10):
    # initialize with random data
    attacker_strategies = [np.random.uniform(size=patrol_problem.wildlife.shape)]
    # TODO: initialize with heuristics
    defender_strategies = np.array([])

    for epoch in range(num_epochs):
        defender_strategy, poacher_strategy = nash_equilibrium(
            defender_strategies,
            attacker_strategies,
        )

    return None
