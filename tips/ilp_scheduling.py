"""An integer linear program that schedules a sports season."""
from abc import ABC
from abc import abstractmethod
from collections import defaultdict
from collections.abc import Callable
from collections.abc import Iterable
from dataclasses import dataclass
from dataclasses import replace
from typing import Optional
from typing import TypeVar
import itertools

from ortools.linear_solver import pywraplp

Team = str
K = TypeVar("K")
V = TypeVar("V")


@dataclass(frozen=True)
class Game:
    week: int  # the 0-indexed week the game occurs
    home_team: Team
    away_team: Team

    def __str__(self):
        return f"{self.week}:{self.away_team}@{self.home_team}"


Schedule = list[Game]


class PreferenceRule(ABC):
    """A PreferenceRule represents a 'would like' constraint in the scheduler.

    The rule is responsible for modifying the model by adding necessary
    variables and constraints, and then returns a set of variables and a
    penalty to be optimized for in the solver objective.
    """

    @abstractmethod
    def game_subset_generator(self, all_games: Iterable[Game]) -> Iterable[list[Game]]:
        """Enumerate subsets of games, each of which requires a call to build_model."""
        ...

    @abstractmethod
    def build_model(
        self, solver, game_vars: dict[Game, pywraplp.Variable],
    ) -> pywraplp.Variable:
        """Make necessary model changes for one subset of games, and output a
        single variable to be included in the objective. This variable is
        interpreted as a count or severity of violating the rule, and will be
        minimized.

        Note: mutates the input `solver` object.
        """
        ...

    @abstractmethod
    def objective_penalty(self) -> int:
        """Return the penalty for violating the preference rule."""
        ...


def partition_by(
    d: dict[Game, V],
    key: Optional[Callable[[Game], K]] = None,
    keys: Optional[Iterable[Callable[[Game], K]]] = None,
) -> dict[K, dict[Game, V]]:
    """Partitions the given Game dict into sub dictionaries using the given keys.

    Values from multiple keys are merged in the final output dict."""
    partitions: defaultdict[K, dict[Game, V]] = defaultdict(dict)
    keys = list(keys) if keys else []
    if key:
        keys += [key]
    for g, v in d.items():
        for key in keys:
            partitions[key(g)][g] = v
    return dict(partitions)


def optimal_schedule(
    weeks: int,
    matchups: Iterable[tuple[Team, Team]],
    rules: Optional[list[PreferenceRule]] = None,
) -> Schedule:
    # all_teams = list(itertools.chain(*matchups))
    all_games = [
        Game(week, home_team=m[0], away_team=m[1])
        for (week, m) in itertools.product(range(weeks), matchups)
    ]
    # include opposite choice for home vs away games.
    all_games += [
        replace(g, home_team=g.away_team, away_team=g.home_team) for g in all_games
    ]

    solver = pywraplp.Solver.CreateSolver("SCIP")  # the default OR-tools ILP solver
    game_vars = {g: solver.IntVar(0, 1, str(g)) for g in all_games}

    # generate indices for efficient model building
    vars_by_team = partition_by(
        game_vars, keys=[lambda g: g.home_team, lambda g: g.away_team],
    )
    vars_by_matchup = partition_by(
        game_vars, key=lambda g: tuple(sorted([g.home_team, g.away_team])),
    )

    # Exactly one game for a matchup is chosen.
    for matchup, var_dict in vars_by_matchup.items():
        # only one matchup per team pair is supported
        assert len(var_dict) == weeks * 2
        solver.Add(sum(var_dict.values()) == 1)

    # A team may play at most one game per week (some weeks are "bye"s)
    for team, team_vars in vars_by_team.items():
        week_to_vars = partition_by(team_vars, key=lambda g: g.week)
        for week, week_vars in week_to_vars.items():
            solver.Add(sum(week_vars.values()) <= 1)

    objective = solver.Objective()
    objective.SetMinimization()
    penalty_vars = []
    rules = rules or []
    for rule in rules:
        print(f"Building models for {rule.__class__.__name__}")
        var_count = 0
        for subset in rule.game_subset_generator(all_games):
            subset_vars = rule.build_model(solver, {g: game_vars[g] for g in subset})
            var_count += len(subset_vars)
            penalty_vars.extend(subset_vars)
            for var in subset_vars:
                objective.SetCoefficient(var, rule.objective_penalty())
        print(f"Generated {var_count} vars for rule {rule.__class__.__name__}.")

    status = solver.Solve()

    if status not in [solver.OPTIMAL, solver.FEASIBLE]:
        raise ValueError("Unable to find feasible solution")

    for var in penalty_vars:
        if var.solution_value() > 0:
            print(f"Violated preference rule {var.solution_value()}={var}")

    return [g for (g, g_var) in game_vars.items() if g_var.solution_value() == 1]


class MinimizeThreeAwayGames(PreferenceRule):
    """A rule that incentivizes against a team having three away games on
    consecutive weeks."""

    def __init__(self, penalty: int = 10):
        self.penalty = penalty

    def game_subset_generator(self, all_games: Iterable[Game]) -> Iterable[list[Game]]:
        """For each week and team, yield the list of away games for that week
        and the subsequent two weeks."""
        # index by away team and week
        index: defaultdict[int, defaultdict[Team, list[Game]]] = defaultdict(
            lambda: defaultdict(list),
        )
        for game in all_games:
            index[game.week][game.away_team].append(game)

        last_week = max(index.keys())
        for week, team_to_games in index.items():
            if week <= last_week - 2:  # [week=last_week-2, last_week-1, last_week]
                for team, games in team_to_games.items():
                    yield games + index[week + 1][team] + index[week + 2][team]

    def build_model(self, solver, game_vars) -> Iterable[list[pywraplp.Variable]]:
        """Create a new variable that is constrained to be 1 if there are three
        away games in the given variables. Note a separate constraint enforces
        that only one game is played per (team, week). This indicates an
        instance of a violation of the "no three away games in a row" rule."""
        which_weeks = list(sorted(list({str(g.week) for g in game_vars})))
        team = next(iter(game_vars.keys())).away_team
        assert len(which_weeks) == 3  # or else subset generator is broken
        violation_var = solver.IntVar(
            0, 1, f"ThreeAwayGames_{team}_{','.join(which_weeks)}",
        )

        # A modeling technique for an implication. The RHS is at most three due
        # to the contraint that each team plays at most one game per week. And
        # if it is equal to 3, then violation_var must be equal to 1. Because
        # the objective minimizes the penalties, when the LHS is less than 3,
        # the violation_var can be set to zero, saving on the objective without
        # affecting anything else in the model.
        solver.Add(sum(game_vars.values()) <= violation_var + 2)
        return [violation_var]

    def objective_penalty(self) -> int:
        return self.penalty


class NoFarTravel(PreferenceRule):
    """A rule that incentivizes against a team having to travel far in two
    consecutive weeks."""

    def __init__(self, far_pairs: Iterable[tuple[Team, Team]], penalty: int = 1):
        self.far_pairs = {tuple(sorted(x)) for x in far_pairs}
        self.penalty = penalty

    def game_subset_generator(self, all_games: Iterable[Game]) -> Iterable[list[Game]]:
        """For each week and team, yield the list of games for that week
        and the subsequent week."""
        # index by week and team
        index: defaultdict[int, defaultdict[Team, list[Game]]] = defaultdict(
            lambda: defaultdict(list),
        )
        for game in all_games:
            index[game.week][game.away_team].append(game)
            index[game.week][game.home_team].append(game)

        last_week = max(index.keys())
        for week, team_to_games in index.items():
            if week <= last_week - 1:
                for team, games in team_to_games.items():
                    yield games + index[week + 1][team]

    def build_model(self, solver, game_vars) -> Iterable[list[pywraplp.Variable]]:
        """Create new variables constrained to be 1 if a two games in subsequent
        weeks involve traveling far."""
        which_weeks = list(sorted(list({g.week for g in game_vars})))
        assert len(which_weeks) == 2  # or else subset generator is broken
        week1, week2 = which_weeks

        # The input game_vars all involve a common team, so the most common
        # team occurring in the list is the team to use for later checks.
        teams = [g.home_team for g in game_vars]
        team = max(set(teams), key=teams.count)

        week_to_vars = partition_by(game_vars, key=lambda g: g.week)

        violation_vars = []
        for g1, g2 in itertools.product(week_to_vars[week1], week_to_vars[week2]):
            # `team` is always playing at Game.home_team
            sites = tuple(sorted([g1.home_team, g2.home_team]))
            if sites in self.far_pairs:
                violation_var = solver.IntVar(
                    0,
                    1,
                    (
                        f"NoFarTravel_{team}_{g1.home_team},{g2.home_team}"
                        f"_{','.join([str(x) for x in which_weeks])}"
                    ),
                )
                violation_vars.append(violation_var)
                # Similar modeling trick to MinimizeThreeAwayGames:
                # violation_var is 1 if and only if both RHS vars are 1.
                solver.Add(game_vars[g1] + game_vars[g2] <= violation_var + 1)

        return violation_vars

    def objective_penalty(self) -> int:
        return self.penalty
