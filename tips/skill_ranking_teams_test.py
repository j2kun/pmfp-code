from skill_ranking_teams import Player
from skill_ranking_teams import Rating
from skill_ranking_teams import Team
from skill_ranking_teams import update_ratings

p1 = Player(1)
p2 = Player(2)
p3 = Player(3)
p4 = Player(4)


def test_unsurprising_t1_win():
    p1_rating = Rating(mean=10, stddev=1)
    p2_rating = Rating(mean=11, stddev=1)
    p3_rating = Rating(mean=2, stddev=1)
    p4_rating = Rating(mean=3, stddev=1)

    team1 = Team(
        ratings={
            p1: p1_rating,
            p2: p2_rating,
        }
    )
    team2 = Team(
        ratings={
            p3: p3_rating,
            p4: p4_rating,
        }
    )

    outcome = 1
    new_ratings = update_ratings(team1, team2, outcome)

    new_p1_rating = new_ratings[p1]
    new_p2_rating = new_ratings[p2]
    new_p3_rating = new_ratings[p3]
    new_p4_rating = new_ratings[p4]

    assert new_p1_rating.mean > p1_rating.mean
    assert new_p2_rating.mean > p2_rating.mean
    assert new_p3_rating.mean < p3_rating.mean
    assert new_p4_rating.mean < p4_rating.mean


def test_surprising_t2_win():
    p1_rating = Rating(mean=10, stddev=0.1)
    p2_rating = Rating(mean=11, stddev=0.1)
    p3_rating = Rating(mean=2, stddev=0.1)
    p4_rating = Rating(mean=3, stddev=0.1)

    team1 = Team(
        ratings={
            p1: p1_rating,
            p2: p2_rating,
        }
    )
    team2 = Team(
        ratings={
            p3: p3_rating,
            p4: p4_rating,
        }
    )

    outcome = -1
    new_ratings = update_ratings(team1, team2, outcome)

    new_p1_rating = new_ratings[p1]
    new_p2_rating = new_ratings[p2]
    new_p3_rating = new_ratings[p3]
    new_p4_rating = new_ratings[p4]

    assert new_p1_rating.mean < p1_rating.mean
    assert new_p2_rating.mean < p2_rating.mean
    assert new_p3_rating.mean > p3_rating.mean
    assert new_p4_rating.mean > p4_rating.mean
    assert new_p1_rating.stddev > p1_rating.stddev
    assert new_p2_rating.stddev > p2_rating.stddev
    assert new_p3_rating.stddev > p3_rating.stddev
    assert new_p4_rating.stddev > p4_rating.stddev
    assert new_p1_rating.mean + new_p2_rating.mean < p3_rating.mean + p4_rating.mean


def test_team_asymmetry():
    p1_rating = Rating(mean=10, stddev=0.1)
    p2_rating = Rating(mean=6, stddev=0.1)
    p3_rating = Rating(mean=2, stddev=0.1)
    p4_rating = Rating(mean=3, stddev=0.1)

    team1 = Team(
        ratings={
            p1: p1_rating,
        }
    )
    team2 = Team(
        ratings={
            p2: p2_rating,
            p3: p3_rating,
            p4: p4_rating,
        }
    )

    outcome = -1
    new_ratings = update_ratings(team1, team2, outcome)

    # team2 is just barely better than team1, so the win is not surprising.

    new_p1_rating = new_ratings[p1]
    new_p2_rating = new_ratings[p2]
    new_p3_rating = new_ratings[p3]
    new_p4_rating = new_ratings[p4]

    assert new_p1_rating.mean < p1_rating.mean
    assert new_p2_rating.mean > p2_rating.mean
    assert new_p3_rating.mean > p3_rating.mean
    assert new_p4_rating.mean > p4_rating.mean

    # p3's rating goes up more due to having a bigger share of the total
    # prior skill contribution to the team
    assert new_p2_rating.mean - p2_rating.mean > new_p3_rating.mean - p3_rating.mean
    assert new_p2_rating.mean - p2_rating.mean > new_p4_rating.mean - p4_rating.mean


def test_team_asymmetry_draw():
    p1_rating = Rating(mean=10, stddev=0.1)
    p2_rating = Rating(mean=6, stddev=0.1)
    p3_rating = Rating(mean=2, stddev=0.1)
    p4_rating = Rating(mean=3, stddev=0.1)

    team1 = Team(
        ratings={
            p1: p1_rating,
        }
    )
    team2 = Team(
        ratings={
            p2: p2_rating,
            p3: p3_rating,
            p4: p4_rating,
        }
    )

    outcome = 0
    new_ratings = update_ratings(team1, team2, outcome)

    # team2 is just barely better than team1, so the draw is evidence p1 is
    # better.

    new_p1_rating = new_ratings[p1]
    new_p2_rating = new_ratings[p2]
    new_p3_rating = new_ratings[p3]
    new_p4_rating = new_ratings[p4]

    assert new_p1_rating.mean > p1_rating.mean
    assert new_p2_rating.mean < p2_rating.mean
    assert new_p3_rating.mean < p3_rating.mean
    assert new_p4_rating.mean < p4_rating.mean

    # p3's rating goes down more due to having a bigger share of the total
    # prior skill contribution to the team
    assert abs(new_p2_rating.mean - p2_rating.mean) > abs(
        new_p3_rating.mean - p3_rating.mean
    )
    assert abs(new_p2_rating.mean - p2_rating.mean) > abs(
        new_p4_rating.mean - p4_rating.mean
    )


def test_huge_upset_via_win():
    # this test exists to test the boundaries of the CDF/PDF functions
    # and the limiting formula to handle them.
    p1_rating = Rating(mean=2, stddev=0.1)
    p2_rating = Rating(mean=2, stddev=0.1)
    p3_rating = Rating(mean=10, stddev=0.1)
    p4_rating = Rating(mean=100, stddev=0.1)

    team1 = Team(
        ratings={
            p1: p1_rating,
            p2: p2_rating,
        }
    )
    team2 = Team(
        ratings={
            p3: p3_rating,
            p4: p4_rating,
        }
    )

    outcome = 1
    new_ratings = update_ratings(team1, team2, outcome)

    new_p1_rating = new_ratings[p1]
    new_p2_rating = new_ratings[p2]
    new_p3_rating = new_ratings[p3]
    new_p4_rating = new_ratings[p4]

    assert new_p1_rating.mean > p1_rating.mean
    assert new_p2_rating.mean > p2_rating.mean
    assert new_p3_rating.mean < p3_rating.mean
    assert new_p4_rating.mean < p4_rating.mean


def test_huge_upset_via_draw():
    # this test exists to test the boundaries of the CDF/PDF functions
    # and the limiting formula to handle them.
    p1_rating = Rating(mean=2, stddev=0.1)
    p2_rating = Rating(mean=2, stddev=0.1)
    p3_rating = Rating(mean=10, stddev=0.1)
    p4_rating = Rating(mean=100, stddev=0.1)

    team1 = Team(
        ratings={
            p1: p1_rating,
            p2: p2_rating,
        }
    )
    team2 = Team(
        ratings={
            p3: p3_rating,
            p4: p4_rating,
        }
    )

    outcome = 0
    new_ratings = update_ratings(team1, team2, outcome)

    new_p1_rating = new_ratings[p1]
    new_p2_rating = new_ratings[p2]
    new_p3_rating = new_ratings[p3]
    new_p4_rating = new_ratings[p4]

    assert new_p1_rating.mean > p1_rating.mean
    assert new_p2_rating.mean > p2_rating.mean
    assert new_p3_rating.mean < p3_rating.mean
    assert new_p4_rating.mean < p4_rating.mean
