"""An implementation of a patrol scheduling algorithm.

Similar to what is used in the PAWS project, see https://github.com/lily-x/mirror and
https://arxiv.org/abs/2106.08413
References to "the paper" and equation markers are in reference to that paper,
v1 submitted 2021-06-15.
"""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import generic_filter
from scipy.stats import logistic


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
class PoacherParameters:
    # A 2D array of floats representing how attractive a grid cell is for
    # poachers to attack. Referred to as z_i, and the entire set as Z. See
    # equation (5).
    attractiveness: np.ndarray


def build_attack_model(
    poacher_parameters: PoacherParameters,
    patrol_problem: PatrolProblem,
    patrol_effort: np.ndarray,
) -> np.ndarray:
    footprint = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    # computes the sum of values of neighbors of each cell, using zero for
    # values that go outside the boundary of the array.
    neighbor_effort = generic_filter(
        patrol_effort,
        np.sum,
        footprint=footprint,
        mode="constant",
        cval=0,
    )
    logistic_input = (
        poacher_parameters.attractiveness
        + patrol_problem.return_on_effort * patrol_effort
        + patrol_problem.displacement_effect * neighbor_effort
    )
    return logistic.cdf(logistic_input)


def sample_poacher_activity(attack_model: np.ndarray) -> np.ndarray:
    return np.random.binomial(1, attack_model)


def update_wildlife(
    wildlife: np.ndarray,
    poacher_activity: np.ndarray,
    patrol_effort: np.ndarray,
    patrol_problem: PatrolProblem,
) -> np.ndarray:
    manmade_change = (
        patrol_problem.poacher_strength * poacher_activity * (1 - patrol_effort)
    )
    natural_growth = wildlife**patrol_problem.wildlife_growth_ratio
    return np.maximum(0, natural_growth - manmade_change)


def schedule_patrols():
    return None
