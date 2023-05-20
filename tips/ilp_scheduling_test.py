from collections import defaultdict
import itertools

from hypothesis import given
from hypothesis import settings
from hypothesis.strategies import composite
from hypothesis.strategies import integers
from hypothesis.strategies import lists
from hypothesis.strategies import permutations
from hypothesis.strategies import text
import pytest

from tips.ilp_scheduling import optimal_schedule
from tips.ilp_scheduling import MinimizeThreeAwayGames
from tips.ilp_scheduling import NoFarTravel


CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


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


TEAMS = ["DAL", "NYG", "PHI", "A", "B"]
MATCHUPS = list(itertools.combinations(TEAMS, 2))
FAR_PAIRS = [
    ("A", "DAL"),
    ("A", "PHI"),
    ("A", "NYG"),
    ("B", "DAL"),
    ("B", "PHI"),
    ("B", "NYG"),
]


def teams_violating_travel(solution, far_pairs=FAR_PAIRS):
    games_by_week = defaultdict(dict)
    for game in solution:
        games_by_week[game.week][game.home_team] = game
        games_by_week[game.week][game.away_team] = game

    violations = defaultdict(set)
    for week, team_to_game in dict(games_by_week).items():
        for team, game1 in team_to_game.items():
            location1 = game1.home_team
            if team in games_by_week[week + 1]:
                game2 = games_by_week[week + 1][team]
                location2 = game2.home_team
                pair = tuple(sorted((location1, location2)))
                if pair in far_pairs:
                    violations[team].add((game1, game2))

    return violations


def teams_violating_three_away_game(solution):
    away_games_by_team = defaultdict(set)
    num_weeks = 0
    for game in solution:
        away_games_by_team[game.away_team].add(game.week)
        num_weeks = max(num_weeks, game.week)

    possible_failing_week_triples = set(
        [(i, i + 1, i + 2) for i in range(num_weeks - 2)]
    )
    violations = dict()
    for team, away_game_weeks in away_games_by_team.items():
        triples_of_away_game_weeks = itertools.combinations(sorted(away_game_weeks), 3)
        bad_triples = possible_failing_week_triples & set(triples_of_away_game_weeks)
        if bad_triples:
            violations[team] = bad_triples

    return violations


def test_preferences_covers_all_matchups_and_avoids_penalties():
    # In this test, there are 6 weeks, which gives the solver
    # enough flexibility to avoid all penalties.
    num_weeks = 6
    solution = optimal_schedule(
        num_weeks,
        MATCHUPS,
        rules=[
            MinimizeThreeAwayGames(penalty=10),
            NoFarTravel(far_pairs=FAR_PAIRS, penalty=10),
        ],
    )
    covered_matchups = [(game.home_team, game.away_team) for game in solution]
    assert sort_and_sort(MATCHUPS) == sort_and_sort(covered_matchups)

    for team, violations in teams_violating_three_away_game(solution).items():
        assert not violations, f"Team {team} violated 'three road games' rule"

    for team, travel_pairs in teams_violating_travel(solution).items():
        assert not travel_pairs, f"Team {team} violated far travel"


def test_preferences_cannot_avoid_all_penalties_violates_travel():
    # 5 weeks is not enough to avoid all penalties, so pick the cheaper penalty
    # to violate: far travel.
    num_weeks = 5
    solution = optimal_schedule(
        num_weeks,
        MATCHUPS,
        rules=[
            MinimizeThreeAwayGames(penalty=10),
            NoFarTravel(far_pairs=FAR_PAIRS, penalty=1),
        ],
    )

    away_game_violations = teams_violating_three_away_game(solution)
    for team, violations in away_game_violations.items():
        assert not violations, f"Team {team} violated 'three road games' rule"

    travel_violations = teams_violating_travel(solution)
    assert travel_violations != dict()
    assert len(travel_violations) == 2


def test_violated_penalty_follows_penalty_magnitude():
    # 5 weeks is not enough to avoid all penalties, so pick the cheaper penalty
    # to violate: triples of away games. Far travel is impossible to avoid
    # completely, so settle for reducing one travel violation in exchange for
    # one triple of away games violation.
    solution = optimal_schedule(
        5,
        MATCHUPS,
        rules=[
            MinimizeThreeAwayGames(penalty=1),
            NoFarTravel(far_pairs=FAR_PAIRS, penalty=10),
        ],
    )

    away_game_violations = teams_violating_three_away_game(solution)
    assert away_game_violations != dict()
    assert len(away_game_violations) == 1

    travel_violations = teams_violating_travel(solution)
    assert travel_violations != dict()
    assert len(travel_violations) == 1


@composite
def random_matchups(draw, min_teams=4, max_teams=10, max_matchups=30):
    """Generate a random set of matchups."""
    team_names = lists(
        elements=text(alphabet=CHARS, min_size=3, max_size=3),
        min_size=min_teams,
        max_size=max_teams,
        unique=True,
    )
    teams = draw(team_names)
    team_pairs = list(itertools.combinations(teams, 2))
    matchups = draw(permutations(team_pairs))
    truncate_pt = draw(
        integers(min_value=4, max_value=min(max_matchups, len(matchups)))
    )
    truncated_matchups = matchups[:truncate_pt]

    far_pairs = draw(permutations(team_pairs))
    truncate_pt = draw(integers(min_value=4, max_value=len(far_pairs)))
    truncated_far_pairs = far_pairs[:truncate_pt]

    return truncated_matchups, truncated_far_pairs


@pytest.mark.order(index=1)
@settings(deadline=100000)
@given(random_matchups())
def test_smoke_feasibility(matchups_and_pairs):
    matchups, far_pairs = matchups_and_pairs

    # only expecting it doesn't raise due to infeasibility
    optimal_schedule(
        len(matchups),  # plenty of weeks
        matchups,
        rules=[
            MinimizeThreeAwayGames(),
            NoFarTravel(far_pairs=far_pairs),
        ],
    )
