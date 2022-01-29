'''An implementation of Elo skill ranking.'''

from dataclasses import dataclass
from math import erf
from math import sqrt
from typing import Tuple


@dataclass
class EloSkill:
    mean: float
    variance: float


def standard_normal_cumulative_density(x):
    '''Cumulative density function for a normal distribution with zero mean and variance 1.'''
    return (1.0 + erf(x / sqrt(2.0))) / 2.0


def elo_player1_win_prob(e1: EloSkill, e2: EloSkill):
    return standard_normal_cumulative_density(
        (e1.mean - e2.mean) / sqrt(e1.variance + e2.variance))


def truncate(x, xmin: int = 0, xmax: int = 3000):
    return min(max(x, xmin), xmax)


def elo_update(
    e1: EloSkill, e2: EloSkill, outcome: int, alpha: float
) -> Tuple[EloSkill, EloSkill]:
    """Update the EloSkills of two players based on a game outcome.

    Args:
      - e1: the EloSkill of player one
      - e2: the EloSkill of player two
      - outcome: an int, 1 if player one wins, -1 if player two wins, and 0 if
        a draw.
      - alpha: a constant describing how much to factor a game into the score
        change.

    Returns:
      A pair (EloSkill, EloSkill) containing the updated skill ratings
    """
    if e1.variance != e2.variance:
        raise ValueError(
            f"Variances must agree, but were "
            f"p1={e1.variance}, p2={e2.variance}")

    std_dev = sqrt(e1.variance)
    scale = alpha * std_dev  # also called the K-factor
    deviation_from_expected = ((outcome + 1) / 2) - elo_player1_win_prob(e1, e2)
    p1_score_change = round(scale * deviation_from_expected)

    return (
        EloSkill(truncate(e1.mean + p1_score_change), e1.variance),
        EloSkill(truncate(e2.mean - p1_score_change), e2.variance),
    )
