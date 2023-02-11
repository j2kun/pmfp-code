import ast
import operator

import pytest

from scalar_evolution import Assign
from scalar_evolution import Increment
from scalar_evolution import Loop
from scalar_evolution import Recurrence
from scalar_evolution import reduce_strength


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
            0,
            Recurrence(7, operator.add, 3),
            Recurrence(7, operator.add, 3),
        ),
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
            0,
            Recurrence(7, operator.add, 3),
            0,
        ),
        (
            1,
            Recurrence(7, operator.add, 3),
            Recurrence(7, operator.add, 3),
        ),
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
    "expression, induction_vars, expected",
    [
        (
            "x*x",
            {"x": Recurrence(0, operator.add, 1)},
            Recurrence(0, operator.add, Recurrence(1, operator.add, 2)),
        ),
        (
            "3*x + 7",
            {"x": Recurrence(0, operator.add, 1)},
            Recurrence(7, operator.add, 3),
        ),
        (
            "x*x*x + 2*x*x + 3*x + 7",
            {"x": Recurrence(0, operator.add, 1)},
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
            {"x": Recurrence(0, operator.add, 2)},  # x += 2
            Recurrence(0, operator.add, 4),  # equivalent to 2x
        ),
    ],
)
def test_rewrite_terms(expression, induction_vars, expected):
    parsed_ast = ast.parse(expression)
    actual = Recurrence.from_ast(parsed_ast, induction_vars)
    assert expected == actual


def test_from_ast_error():
    with pytest.raises(ValueError):
        Recurrence.from_ast(ast.Name(id="z"), induction_vars=dict())


@pytest.mark.parametrize(
    "input, expected",
    [
        (
            Recurrence(0, operator.add, Recurrence(4, operator.add, 0)),
            Recurrence(0, operator.add, 4),
        ),
        (
            Recurrence(0, operator.mul, 3),
            0,
        ),
        (
            Recurrence(7, operator.add, 0),
            7,
        ),
        (
            Recurrence(5, operator.mul, 1),
            5,
        ),
    ],
)
def test_normalize(input, expected):
    actual = input.normalize()
    assert expected == actual


def parse_to_binop(expr: str):
    # Can ignore type checker because input is assumed to be a BinOp, which has
    # the attribute
    return ast.parse(expr).body[0].value  # type:ignore


def test_reduce_strength():
    # first, demonstrate the rewriting is correct
    # original
    y = 0
    for x in range(10):
        y = y + x * x * x + 2 * x * x + 3 * x + 7
    original_y = y

    # rewritten
    t0 = 0
    t1 = 7
    t2 = 6
    t3 = 10
    t4 = 6
    for x in range(10):
        t0 = t0 + t1
        t1 = t1 + t2
        t2 = t2 + t3
        t3 = t3 + t4
        y = t0
    rewritten_y = y

    assert original_y == rewritten_y

    # Then run the test with inputs in the proper AST form
    loop = Loop(
        header=[],
        body=[
            Increment(lhs=ast.Name("y"), rhs=parse_to_binop("x*x*x + 2*x*x + 3*x + 7"))
        ],
        context={
            "x": Recurrence(0, operator.add, 1),
        },
    )
    # The SCEV for x is {7, +, 6, +, 10, + 6}
    # The SCEV for y is {0, +, 7, +, 6, +, 10, + 6}
    expected = Loop(
        header=[
            Assign(lhs=ast.Name(id="t0"), rhs=ast.Constant(value=0)),
            Assign(lhs=ast.Name(id="t1"), rhs=ast.Constant(value=7)),
            Assign(lhs=ast.Name(id="t2"), rhs=ast.Constant(value=6)),
            Assign(lhs=ast.Name(id="t3"), rhs=ast.Constant(value=10)),
            Assign(lhs=ast.Name(id="t4"), rhs=ast.Constant(value=6)),
        ],
        body=[
            Increment(lhs=ast.Name(id="t0"), rhs=ast.Name(id="t1")),
            Increment(lhs=ast.Name(id="t1"), rhs=ast.Name(id="t2")),
            Increment(lhs=ast.Name(id="t2"), rhs=ast.Name(id="t3")),
            Increment(lhs=ast.Name(id="t3"), rhs=ast.Name(id="t4")),
            Assign(lhs=ast.Name(id="y"), rhs=ast.Name(id="t0")),
        ],
        context={},
    )
    assert expected == reduce_strength(loop)


def test_reduce_strength_unsupported_assign():
    loop = Loop(
        header=[],
        body=[Assign(lhs=ast.Name("y"), rhs=parse_to_binop("x*x*x + 2*x*x + 3*x + 7"))],
        context={
            "x": Recurrence(0, operator.add, 1),
        },
    )
    with pytest.raises(ValueError):
        reduce_strength(loop)
