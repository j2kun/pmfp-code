from collections import Counter
from statistics import NormalDist

from skill_ranking_teams import Gaussian
from skill_ranking_teams import two_player_update


def test_two_player_update():
    p1_prior = Gaussian()
    p2_prior = Gaussian()
    outcome = 1

    new_p1, new_p2 = two_player_update(p1_prior, p2_prior, outcome)
    assert new_p1.mean() > p1_prior.mean()
    assert new_p2.mean() < p2_prior.mean()
