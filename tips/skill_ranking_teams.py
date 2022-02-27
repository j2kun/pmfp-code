'''A bare-bones implementation of two-team TrueSkill.'''

from dataclasses import dataclass
from math import sqrt
from statistics import NormalDist
from typing import Dict
from typing import NewType

STANDARD_NORMAL = NormalDist(0, 1)
DEFAULT_MEAN = 25
DEFAULT_STD_DEV = DEFAULT_MEAN / 3
DEFAULT_VARIANCE = DEFAULT_STD_DEV ** 2

SKILL_CLASS_WIDTH = (DEFAULT_STD_DEV / 2) ** 2
DRAW_PROBABILITY = 0.01  # 1 percent
ADDITIVE_DYNAMICS_FACTOR = (DEFAULT_STD_DEV / 100) ** 2
TOLERANCE = 1e-18


# players are globally unique integer ids
Player = NewType('Player', int)


@dataclass(frozen=True)
class Rating:
    mean: float = DEFAULT_MEAN
    stddev: float = DEFAULT_STD_DEV


@dataclass(frozen=True)
class Team:
    # A team may have one or more players
    ratings: Dict[Player, Rating]


def compute_draw_margin() -> float:
    '''
      The margin to use to consider the game a draw, based on the (pre-set)
      probability of a draw, which is typically set by measuring the draw rates
      of a large number of games. Derived by inverting the formula

        P(draw) = -1 + 2 * normal_cdf(
            draw_margin / sqrt(player_count * skill_class_width)
        )

      or, as written exactly in the paper,

        P(draw) = -1 + 2 * normal_cdf(
            draw_margin / (sqrt(n1 + n2) * beta)
        )
    '''
    inv_cdf_arg = 0.5 * (DRAW_PROBABILITY + 1)
    inv_cdf_output = STANDARD_NORMAL.inv_cdf(inv_cdf_arg)
    return inv_cdf_output * sqrt(2 * SKILL_CLASS_WIDTH)


def truncated_onesided_gaussian_v(t: float, lower: float) -> float:
    '''Equation 4.7 from Herbrich '05, On Gaussian Expectation Propagation.

    Representing the additive correction factor to the mean of a rectified
    Gaussian that is only truncated from below.

    This function grows roughly like lower - t for t < lower and quickly
    approaches zero for t > lower.
    '''
    normalization = STANDARD_NORMAL.cdf(t - lower)
    if normalization < TOLERANCE:
        return lower - t   # approaches infinity
    return STANDARD_NORMAL.pdf(t - lower) / normalization


def truncated_onesided_gaussian_w(t: float, lower: float) -> float:
    '''Equation 4.8 from Herbrich '05, On Gaussian Expectation Propagation.

    Representing the additive correction factor to the variance of a rectified
    Gaussian that is only truncated from below. This is a smooth approximation
    to an indicator function for the condition t <= lower.
    '''
    v_value = truncated_onesided_gaussian_v(t, lower)
    return v_value * (v_value + t - lower)


def truncated_twosided_gaussian_v(t, lower, upper):
    '''Equation 4.4 from Herbrich '05, On Gaussian Expectation Propagation.

    Representing the additive correction factor to the mean of a rectified
    Gaussian that is truncated on both sides.
    '''
    normalization = (STANDARD_NORMAL.cdf(upper - t) - STANDARD_NORMAL.cdf(lower - t))
    if normalization < TOLERANCE:
        # the limit as upper -> lower
        return lower - t

    return (
        STANDARD_NORMAL.pdf(lower - t) - STANDARD_NORMAL.pdf(upper - t)
    ) / normalization


def truncated_twosided_gaussian_w(perf_diff, draw_margin):
    '''Equation 4.4 from Herbrich '05, On Gaussian Expectation Propagation.

    Representing the additive correction factor to the variance of a rectified
    Gaussian that is truncated on both sides.
    '''
    abs_diff = abs(perf_diff)
    normalization = (
        STANDARD_NORMAL.cdf(draw_margin - abs_diff) - STANDARD_NORMAL.cdf(-draw_margin - abs_diff)
    )
    if normalization < TOLERANCE:
        return 1

    v_value = truncated_twosided_gaussian_v(
        perf_diff, -draw_margin, draw_margin)
    t1 = (draw_margin - abs_diff) * STANDARD_NORMAL.pdf(draw_margin - abs_diff)
    t2 = (-draw_margin - abs_diff) * \
        STANDARD_NORMAL.pdf(-draw_margin - abs_diff)
    return v_value ** 2 + (t1 - t2) / normalization


def update_one_team(
        team1: Team, team2: Team, outcome: int) -> Dict[Player, Rating]:
    '''Return the new ratings for team1.'''
    # Each team is treated as if it were a player whose skill is the sum of the
    # skills of individual teammates, and whose variance is the sum of
    # variances of individual teammates. The normalization is adjusted based on
    # the number of players.
    draw_margin = compute_draw_margin()
    player_count = len(team1.ratings) + len(team2.ratings)
    t1_mean = sum(p.mean for p in team1.ratings.values())
    t2_mean = sum(p.mean for p in team2.ratings.values())
    t1_variance = sum(p.stddev ** 2 for p in team1.ratings.values())
    t2_variance = sum(p.stddev ** 2 for p in team2.ratings.values())

    c = sqrt(t1_variance + t2_variance + player_count * SKILL_CLASS_WIDTH)
    winning_mean = t1_mean if outcome >= 0 else t2_mean
    losing_mean = t2_mean if outcome >= 0 else t1_mean
    perf_diff = winning_mean - losing_mean

    # This is where the code sample differs from the text in support for draws.
    # If the outcome is zero, it's a draw, and we must use special "two-sided"
    # Gaussian truncation formulas representing the fact that the difference in
    # performance values was close to zero.
    if outcome == 0:
        v = truncated_twosided_gaussian_v(
            perf_diff / c, -draw_margin / c, draw_margin / c)
        w = truncated_twosided_gaussian_w(perf_diff / c, draw_margin / c)
        mean_adjustment_direction = 1
    else:
        v = truncated_onesided_gaussian_v(perf_diff / c, draw_margin / c)
        w = truncated_onesided_gaussian_w(perf_diff / c, draw_margin / c)
        mean_adjustment_direction = outcome

    # Here we propagate the rating adjustment data from the team-wide summed
    # skills down to each player. The normalization constant c is scaled up
    # according to the sum of variances and player counts, which impacts both
    # the value of c and the size of the multiplier that attributes team
    # performance back to the individual player.
    new_ratings: Dict[Player, Rating] = dict()
    for player, rating in team1.ratings.items():
        mean, stddev = rating.mean, rating.stddev
        mean_multiplier = (mean ** 2 + ADDITIVE_DYNAMICS_FACTOR) / c
        variance_plus_dynamics = stddev ** 2 + ADDITIVE_DYNAMICS_FACTOR
        stddev_multiplier = variance_plus_dynamics / (c ** 2)
        new_mean = mean + mean_adjustment_direction * mean_multiplier * v
        new_stddev = sqrt(variance_plus_dynamics * (1 - w * stddev_multiplier))
        new_ratings[player] = Rating(mean=new_mean, stddev=new_stddev)

    return new_ratings


def update_ratings(team1: Team, team2: Team, outcome: int = 1) -> Dict[Player, Rating]:
    # Nb: in Python3.9 the bitwise-or operator is overloaded for dictionaries
    # to perform a union of the key-value pairs. See PEP584.
    return (
        update_one_team(team1, team2, outcome) | update_one_team(team2, team1, -outcome)
    )
