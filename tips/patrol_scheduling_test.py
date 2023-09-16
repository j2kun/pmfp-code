import numpy as np

from tips.patrol_scheduling import (
    PatrolProblem,
    PoacherParameters,
    build_attack_model,
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
    poacher_parameters = PoacherParameters(
        attractiveness=attractiveness,
    )

    attack_prob = build_attack_model(
        poacher_parameters=poacher_parameters,
        patrol_problem=problem,
        patrol_effort=np.zeros(shape),
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
        patrol_effort=patrol_effort,
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
