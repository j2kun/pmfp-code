import operator

import numpy as np
import pytest

from tips.patrol_scheduling import (
    DefenderPolicy,
    PatrolProblem,
    PoacherPolicy,
    build_attack_model,
    defender_best_response,
    simulate_game,
    update_wildlife,
)


def test_attack_model():
    shape = (5, 5)
    # fmt: off
    attractiveness = np.array([
        -9, -9, -9, -9, -9,
        -9, -9, -9, -9, -9,
        -9, -9,  1,  1, -9,
        -9, -9,  1,  2, -9,
        -9, -9, -9, -9, -9,
    ]).reshape(shape)
    # fmt: on

    problem = PatrolProblem(
        wildlife=np.zeros(shape),
        total_budget=1,
        wildlife_growth_ratio=1.02,
        poacher_strength=0.9,
        return_on_effort=-0.1,
        displacement_effect=0.1,
    )
    poacher_policy = PoacherPolicy(
        attractiveness=attractiveness,
    )
    patrol_effort = DefenderPolicy(
        patrol_effort=np.zeros(shape),
    )

    attack_prob = build_attack_model(
        poacher_policy=poacher_policy,
        patrol_problem=problem,
        defender_policy=patrol_effort,
    )
    # fmt: off
    expected_probs = np.array([
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, .73, .73, 0,
        0, 0, .73, .88, 0,
        0, 0, 0, 0, 0,
    ]).reshape(shape)
    # fmt: on
    np.testing.assert_allclose(expected_probs, attack_prob, atol=1e-2, rtol=1e-2)


def test_attack_model_concentrated():
    shape = (5, 5)
    # fmt: off
    attractiveness = np.array([
        -10, -10, -10, -10, -10,
        -10, -10, -10, -10, -10,
        -10, -10, -10, -10, -10,
        -10, -10, -10,  10, -10,
        -10, -10, -10, -10, -10,
    ]).reshape(shape)
    # fmt: on

    problem = PatrolProblem(
        wildlife=np.zeros(shape),
        total_budget=1,
        wildlife_growth_ratio=1.02,
        poacher_strength=0.9,
        return_on_effort=-0.1,
        displacement_effect=0.1,
    )
    poacher_policy = PoacherPolicy(
        attractiveness=attractiveness,
    )
    patrol_effort = DefenderPolicy(
        patrol_effort=np.zeros(shape),
    )

    attack_prob = build_attack_model(
        poacher_policy=poacher_policy,
        patrol_problem=problem,
        defender_policy=patrol_effort,
    )
    # fmt: off
    expected_probs = np.array([
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 1, 0,
        0, 0, 0, 0, 0,
    ]).reshape(shape)
    # fmt: on
    np.testing.assert_allclose(expected_probs, attack_prob, atol=1e-2, rtol=1e-2)


def test_wildlife_update_no_patrolling():
    shape = (5, 5)
    # fmt: off
    wildlife = np.array([
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 7, 7, 7,
        0, 0, 7, 9, 7,
        0, 0, 7, 7, 7,
    ]).reshape(shape)
    snares = np.array([
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 1, 0, 0,
        0, 0, 1, 1, 0,
        0, 0, 0, 0, 0,
    ]).reshape(shape)
    patrol_effort = np.array([
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
    ]).reshape(shape)
    # fmt: on

    problem = PatrolProblem(
        wildlife=wildlife,
        total_budget=1,
        wildlife_growth_ratio=1.02,
        poacher_strength=0.9,
        return_on_effort=-0.1,
        displacement_effect=0.1,
    )
    new_wildlife = update_wildlife(
        wildlife=wildlife,
        poacher_activity=snares,
        defender_policy=DefenderPolicy(patrol_effort),
        patrol_problem=problem,
    )
    # fmt: off
    expected_wildlife = np.array([
        0, 0,   0,   0,   0,
        0, 0,   0,   0,   0,
        0, 0, 6.3, 7.2, 7.2,
        0, 0, 6.3, 8.5, 7.2,
        0, 0, 7.2, 7.2, 7.2,
    ]).reshape(shape)
    # fmt: on
    np.testing.assert_allclose(expected_wildlife, new_wildlife, atol=1e-1, rtol=1e-1)


def test_simulate_game_full_defense():
    shape = (5, 5)
    wildlife = np.zeros(shape)

    wildlife[2][2:5] = [7, 7, 7]
    wildlife[3][2:5] = [7, 9, 7]
    wildlife[4][2:5] = [7, 7, 7]

    patrol_problem = PatrolProblem(
        wildlife=wildlife,
        total_budget=9,
        wildlife_growth_ratio=1.02,
        poacher_strength=0.9,
        return_on_effort=-0.9,
        displacement_effect=0.1,
    )

    # poach anywhere there are animals, but don't prioritize
    attractiveness = -10 * np.ones(shape=shape)
    attractiveness[2][2:5] = [0, 0, 0]
    attractiveness[3][2:5] = [0, 0, 0]
    attractiveness[4][2:5] = [0, 0, 0]

    def draw_poacher_policy():
        return PoacherPolicy(attractiveness)

    alloc = np.zeros(shape=shape)
    alloc[2][2:5] = [1, 1, 1]
    alloc[3][2:5] = [1, 1, 1]
    alloc[4][2:5] = [1, 1, 1]

    def draw_defender_policy():
        return DefenderPolicy(alloc)

    planning_horizon = 5
    end_wildlife = simulate_game(
        patrol_problem,
        draw_poacher_policy,
        draw_defender_policy,
        planning_horizon=planning_horizon,
    )
    # With enough effort to guard everything, poachers have no effect
    expected_wildlife = wildlife ** (
        patrol_problem.wildlife_growth_ratio**planning_horizon
    )
    np.testing.assert_allclose(end_wildlife, expected_wildlife, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("defense", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
def test_simulate_game_partial_defense(defense):
    shape = (5, 5)
    wildlife = np.zeros(shape)

    wildlife[2][2:5] = [7, 7, 7]
    wildlife[3][2:5] = [7, 9, 7]
    wildlife[4][2:5] = [7, 7, 7]

    patrol_problem = PatrolProblem(
        wildlife=wildlife,
        total_budget=defense * 9,
        wildlife_growth_ratio=1.02,
        poacher_strength=0.9,
        return_on_effort=-0.9,
        displacement_effect=0.1,
    )

    # poach anywhere there are animals, but don't prioritize
    attractiveness = -10 * np.ones(shape=shape)
    attractiveness[2][2:5] = [0, 0, 0]
    attractiveness[3][2:5] = [0, 0, 0]
    attractiveness[4][2:5] = [0, 0, 0]

    def draw_poacher_policy():
        return PoacherPolicy(attractiveness)

    alloc = np.zeros(shape=shape)
    alloc[2][2:5] = [defense, defense, defense]
    alloc[3][2:5] = [defense, defense, defense]
    alloc[4][2:5] = [defense, defense, defense]

    def draw_defender_policy():
        return DefenderPolicy(alloc)

    planning_horizon = 5
    end_wildlife = simulate_game(
        patrol_problem,
        draw_poacher_policy,
        draw_defender_policy,
        planning_horizon=planning_horizon,
    )
    # all of the tests reduce wildlife by some amount, but not maximally
    unpoached_outcome = wildlife ** (
        patrol_problem.wildlife_growth_ratio**planning_horizon
    )
    print(end_wildlife)
    print(unpoached_outcome)
    epsilon = 1e-08
    np.testing.assert_array_compare(
        operator.le,
        end_wildlife - epsilon,
        unpoached_outcome,
        header="Arrays are not lte-ordered",
        equal_inf=False,
    )

    # counterfactual: no defense
    alloc = np.zeros(shape=shape)

    def draw_defender_policy():
        return DefenderPolicy(alloc)

    planning_horizon = 5
    no_defense_end_wildlife = simulate_game(
        patrol_problem,
        draw_poacher_policy,
        draw_defender_policy,
        planning_horizon=planning_horizon,
    )
    np.testing.assert_array_compare(
        operator.le,
        no_defense_end_wildlife - epsilon,
        end_wildlife,
        header="Arrays are not lte-ordered (no defense)",
        equal_inf=False,
    )


# A default test harness that individual tests override parts of
class BestResponseTest:
    def __init__(self):
        self.shape = (5, 5)
        self.wildlife = np.zeros(self.shape)

        # More realistic settings
        # patrol_problem = PatrolProblem(
        #     wildlife=wildlife,
        #     total_budget=1,
        #     wildlife_growth_ratio=1.02,
        #     poacher_strength=0.9,
        #     return_on_effort=-0.1,
        #     displacement_effect=0.1,
        # )

        # exaggerated for test
        self.patrol_problem = PatrolProblem(
            wildlife=self.wildlife,
            total_budget=1,
            wildlife_growth_ratio=1.0,
            poacher_strength=0.5,
            return_on_effort=-0.5,
            displacement_effect=0.0,
        )

        self.poacher_strategies = [np.zeros(self.shape)]
        self.poacher_distribution = [1.0]
        self.training_rounds = 400
        self.planning_horizon = 10

    def train_and_sample(self):
        learner = defender_best_response(
            patrol_problem=self.patrol_problem,
            poacher_strategies=self.poacher_strategies,
            poacher_distribution=self.poacher_distribution,
            training_rounds=self.training_rounds,
            planning_horizon=self.planning_horizon,
        )

        initial_wildlife = self.patrol_problem.wildlife.flatten()
        initial_effort = np.zeros(shape=self.shape).flatten()
        initial_state = np.concatenate([initial_wildlife, initial_effort, [0]])
        action = learner.select_action(initial_state).reshape(self.shape)
        return action


def test_defender_best_response_point_poaching_strategy():
    test = BestResponseTest()
    test.wildlife[3, 3] = 9

    # the initial poacher strategy is to poach in a single cell
    single_cell_strat = -10 * np.ones(shape=test.shape)
    single_cell_strat[3, 3] = 10.0
    test.poacher_strategies = [single_cell_strat]
    # there is only one initial strategy, so the distribution to use is
    # fully concentrated on that strategy
    test.poacher_distribution = [1.0]

    action = test.train_and_sample()
    assert (
        action[3, 3] > 0.75
    ), f"Patrol should be concentrated on cell (3, 3), but was {action.round(2)}"


def test_defender_best_response_concentrated_poaching_strategy():
    test = BestResponseTest()
    test.wildlife[2][2:5] = [7, 7, 7]
    test.wildlife[3][2:5] = [7, 9, 7]
    test.wildlife[4][2:5] = [7, 7, 7]

    test.patrol_problem = PatrolProblem(
        wildlife=test.wildlife,
        total_budget=9,
        wildlife_growth_ratio=1.01,
        poacher_strength=0.9,
        return_on_effort=-0.9,
        displacement_effect=0.0,
    )

    # poach anywhere there are animals, but don't prioritize
    strat = -10 * np.ones(shape=test.shape)
    strat[2][2:5] = [0, 0, 0]
    strat[3][2:5] = [0, 0, 0]
    strat[4][2:5] = [0, 0, 0]
    test.poacher_strategies = [strat]
    test.poacher_distribution = [1.0]

    action = test.train_and_sample()
    wildlife_section = action[2:5, 2:5]
    assert np.sum(wildlife_section) > 0.75, (
        "Patrol should be concentrated in the neighborhood of cell (3, 3), "
        f"but was {action.round(2)}"
    )
    center_value = wildlife_section[1, 1]
    assert np.max(wildlife_section) == center_value, (
        "Patrol should prioritize cell (3, 3) since it has the most wildlife, "
        f"but was {action.round(2)}"
    )
    wildlife_section[1, 1] = 0
    assert center_value > np.max(wildlife_section), (
        "Patrol should prioritize cell (3, 3) since it has the most wildlife, "
        f"but was {action.round(2)}"
    )
