import random
from collections import Counter
from statistics import NormalDist, mean

from hypothesis import assume, given, settings
from hypothesis.strategies import integers

from tips.skill_ranking import EloSkill, elo_player1_win_prob, elo_update


@settings(deadline=2000)
@given(
    integers(min_value=0, max_value=3000),
    integers(min_value=0, max_value=3000),
)
def test_two_player_tournament(player1_skill_mean, player2_skill_mean):
    # If the skill means are too close, the variance will result in random
    # arbitrary outcomes, weighted toward the outcomes of the most recent games,
    # hard to assert for a test.
    assume(abs(player1_skill_mean - player2_skill_mean) > 50)

    # These settings result in a total K-factor of ~25
    standard_deviation = 50
    alpha = 0.3

    p1 = NormalDist(mu=player1_skill_mean, sigma=standard_deviation)
    p2 = NormalDist(mu=player2_skill_mean, sigma=standard_deviation)

    p1_elo = EloSkill(mean=1500, variance=standard_deviation**2)
    p2_elo = EloSkill(mean=1500, variance=standard_deviation**2)

    num_games = 1000
    outcomes = list(zip(p1.samples(num_games, seed=1), p2.samples(num_games, seed=2)))
    print([int(x > y) for (x, y) in outcomes])
    print(Counter([int(x > y) for (x, y) in outcomes]))

    for p1_perf, p2_perf in outcomes:
        outcome = 1 if p1_perf > p2_perf else -1
        if abs(p1_perf - p2_perf) < 1e-03:
            outcome = 0  # a tie
        new_p1_elo, new_p2_elo = elo_update(p1_elo, p2_elo, outcome, alpha)
        if outcome > 0:
            # checking <= instead of < because scores are truncated at the extreme ends
            assert new_p1_elo.mean >= p1_elo.mean
            assert new_p2_elo.mean <= p2_elo.mean
        if outcome < 0:
            assert new_p2_elo.mean >= p2_elo.mean
            assert new_p1_elo.mean <= p1_elo.mean

        p1_elo, p2_elo = new_p1_elo, new_p2_elo

    should_p1_win = player1_skill_mean > player2_skill_mean
    if should_p1_win:
        assert p1_elo.mean > p2_elo.mean
        assert elo_player1_win_prob(p1_elo, p2_elo) > 0.5
    else:
        assert p1_elo.mean < p2_elo.mean
        assert elo_player1_win_prob(p1_elo, p2_elo) < 0.5


def test_n_player_tournament():
    random.seed(1)
    players = list(range(30))
    player_skills = [1000 + i * 50 for i in players]

    standard_deviation = 50
    alpha = 0.4

    perf_dists = [
        NormalDist(mu=skill, sigma=standard_deviation) for skill in player_skills
    ]
    elos = [EloSkill(mean=1500, variance=standard_deviation**2) for _ in players]

    num_games = 30000
    for _ in range(num_games):
        i = random.choice(players)
        j = random.choice(players)

        outcomes = list(
            zip(
                perf_dists[i].samples(3),
                perf_dists[j].samples(3),
            ),
        )
        for p1_perf, p2_perf in outcomes:
            outcome = 1 if p1_perf > p2_perf else -1
            if abs(p1_perf - p2_perf) < 1e-03:
                outcome = 0  # a tie
            elos[i], elos[j] = elo_update(elos[i], elos[j], outcome, alpha)

    skills_vs_elo = [(s, elo.mean) for (s, elo) in zip(player_skills, elos)]
    inaccuracies = [s - elo for (s, elo) in skills_vs_elo]
    assert mean(inaccuracies) < 200
    assert max(inaccuracies) < 500
