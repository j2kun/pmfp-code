import ast
import operator

import pytest

from scalar_evolution import Recurrence


def test_basic_recurrence_repr():
    rec = Recurrence(1, operator.add, 1)
    assert repr(rec) == "{1, +, 1}"


def test_chain_recurrence_repr():
    rec = Recurrence(1, operator.add, 1)
    rec2 = Recurrence(2, operator.mul, rec)
    assert repr(rec2) == "{2, *, 1, +, 1}"


def test_basic_recurrence_evaluate():
    rec = Recurrence(3, operator.add, 7)
    assert 24 == rec.evaluate(3)


def test_chain_recurrence_evaluate():
    # {7, +, 6, +, 10, +, 6}
    # corresponds to the loop
    # for x in range(n):
    #   value = x^3 + 2x^2 + 3x + 7
    rec = Recurrence(
        7, operator.add, Recurrence(6, operator.add, Recurrence(10, operator.add, 6))
    )
    assert 61 == rec.evaluate(3)


@pytest.mark.parametrize(
    "rec1, rec2, expected",
    [
        (
            12,
            Recurrence(7, operator.add, 3),
            Recurrence(19, operator.add, 3),
        ),
        (
            Recurrence(7, operator.add, 3),
            Recurrence(1, operator.add, 1),
            Recurrence(8, operator.add, 4),
        ),
    ],
)
def test_add(rec1, rec2, expected):
    assert expected == rec1 + rec2


@pytest.mark.parametrize(
    "rec1, rec2, expected",
    [
        (
            12,
            Recurrence(7, operator.add, 3),
            Recurrence(84, operator.add, 36),
        ),
        (
            12,
            Recurrence(7, operator.mul, 3),
            Recurrence(84, operator.mul, 3),
        ),
        (
            Recurrence(0, operator.add, 1),
            Recurrence(0, operator.add, 1),
            Recurrence(0, operator.add, Recurrence(1, operator.add, 2)),
        ),
        (
            Recurrence(7, operator.add, 3),
            Recurrence(2, operator.add, 5),
            Recurrence(14, operator.add, Recurrence(56, operator.add, 30)),
        ),
        (
            Recurrence(0, operator.add, Recurrence(1, operator.add, 2)),
            Recurrence(0, operator.add, 1),
            Recurrence(
                0,
                operator.add,
                Recurrence(1, operator.add, Recurrence(6, operator.add, 6)),
            ),
        ),
    ],
)
def test_mul(rec1, rec2, expected):
    actual = rec1 * rec2
    assert expected == actual


@pytest.mark.parametrize(
    "expression, induction_var, expected",
    [
        (
            "x*x",
            Recurrence(0, operator.add, 1),
            Recurrence(0, operator.add, Recurrence(1, operator.add, 2)),
        ),
        (
            "x*x*x + 2*x*x + 3*x + 7",
            Recurrence(0, operator.add, 1),
            Recurrence(
                7,
                operator.add,
                Recurrence(6, operator.add, Recurrence(10, operator.add, 6)),
            ),
        ),
        # Before normalization was added, note the trailing +0 that is unnecessary
        # (
        #     "(x*x + x) - (x*x - x)",
        #     Recurrence(0, operator.add, 2),  # x += 2
        #     Recurrence(0, operator.add, Recurrence(4, operator.add, 0)),
        # ),
        (
            "(x*x + x) - (x*x - x)",
            Recurrence(0, operator.add, 2),  # x += 2
            Recurrence(0, operator.add, 4),  # equivalent to 2x
        ),
    ],
)
def test_rewrite_terms(expression, induction_var, expected):
    parsed_ast = ast.parse(expression)
    actual = Recurrence.from_ast(parsed_ast, induction_vars={"x": induction_var})
    assert expected == actual


@pytest.mark.parametrize(
    "input, expected",
    [
        (
            Recurrence(0, operator.add, Recurrence(4, operator.add, 0)),
            Recurrence(0, operator.add, 4),
        ),
    ],
)
def test_normalize(input, expected):
    actual = input.normalize()
    assert expected == actual
