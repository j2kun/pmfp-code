from collections import defaultdict
import itertools

import pytest

from ilp_scheduling import optimal_schedule
from ilp_scheduling import MinimizeThreeAwayGames
from ilp_scheduling import NoFarTravel


def sort_and_sort(L):
    return tuple(sorted([tuple(sorted(x)) for x in L]))


def test_no_preferences_covers_all_matchups():
    matchups = list(itertools.combinations(["DAL", "NYG", "PHI", "A", "B"], 2))
    solution = optimal_schedule(5, matchups)
    covered_matchups = [(game.home_team, game.away_team) for game in solution]

    # Because we're checking with multiplicity, this covers the case that the
    # solver picks two instances of the same matchup, say, with different
    # choices of home vs away.
    assert sort_and_sort(matchups) == sort_and_sort(covered_matchups)


def test_too_few_weeks_infeasible():
    matchups = list(itertools.combinations(["DAL", "NYG", "PHI", "A", "B"], 2))
    for i in range(1, 5):
        with pytest.raises(ValueError):
            optimal_schedule(i, matchups, [])


# add hypothesis generator for random matchups, pick large enough number of weeks...

def test_each_team_plays_at_most_once_per_week():
    matchups = list(itertools.combinations(["DAL", "NYG", "PHI", "A", "B"], 2))
    solution = optimal_schedule(5, matchups)
    by_week_and_team = defaultdict(lambda: defaultdict(list))
    for game in solution:
        by_week_and_team[game.home_team][game.week].append(game)
        by_week_and_team[game.away_team][game.week].append(game)

    for team in by_week_and_team:
        for week, games in by_week_and_team[team].items():
            assert len(games) <= 1, f"Team {team} played {len(games)} in week {week}"


def test_preferences_covers_all_matchups():
    matchups = list(itertools.combinations(["DAL", "NYG", "PHI", "A", "B"], 2))
    solution = optimal_schedule(
        5,
        matchups,
        rules=[
            MinimizeThreeAwayGames(),
            NoFarTravel(
                # A and B are close, as are DAL/NYG/PHI
                far_pairs=[
                    ("A", "DAL"),
                    ("A", "PHI"),
                    ("A", "NYG"),
                    ("B", "DAL"),
                    ("B", "PHI"),
                    ("B", "NYG"),
                ]
            ),
        ],
    )
    covered_matchups = [(game.home_team, game.away_team) for game in solution]
    assert sort_and_sort(matchups) == sort_and_sort(covered_matchups)


    # TODO: assert that nobody has 3 away games in a row, and there is no far
    # travel.
