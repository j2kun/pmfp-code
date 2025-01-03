import ast
import operator
from dataclasses import dataclass
from typing import Callable, Union

# only operator.add and operator.mul are supported
Operator = Callable[[int, int], int]
allowed_ops = [operator.add, operator.mul]


@dataclass
class Assign:
    lhs: ast.Name
    rhs: ast.AST

    def __repr__(self):
        return f"{self.lhs.id} = {ast.dump(self.rhs)}"

    def __eq__(self, other):
        return (
            isinstance(other, Assign)
            and self.lhs.id == other.lhs.id
            and ast.dump(self.rhs) == ast.dump(other.rhs)
        )


@dataclass
class Increment(Assign):
    def __repr__(self):
        return f"{self.lhs.id} += {ast.dump(self.rhs)}"

    def __eq__(self, other):
        return (
            isinstance(other, Increment)
            and self.lhs.id == other.lhs.id
            and ast.dump(self.rhs) == ast.dump(other.rhs)
        )


class Recurrence:
    def __init__(
        self,
        base: int,
        op: Operator,
        increment: Union["Recurrence", int],
    ):
        assert op in allowed_ops
        assert isinstance(base, int)
        self.base = base
        self.op = op
        self.increment = increment

    def visit(self, fn):
        fn(self)
        if isinstance(self.increment, int):
            fn(self.increment)
        else:
            self.increment.visit(fn)

    def evaluate(self, i):
        if i == 0:
            return self.base

        if isinstance(self.increment, int):
            increment = self.increment
        else:
            increment = self.increment.evaluate(i - 1)

        return self.op(self.evaluate(i - 1), increment)

    def br_notation(self, flatten=True):
        op_str = "+" if self.op == operator.add else "*"
        nested = repr(self.increment)
        if flatten:
            nested = nested.strip("{").strip("}")
        return f"{{{self.base}, {op_str}, {nested}}}"

    def __repr__(self):
        return self.br_notation(flatten=True)

    def __eq__(self, other):
        return isinstance(other, Recurrence) and repr(self) == repr(other)

    def __hash__(self):
        return hash((self.base, self.op, self.increment))

    def __radd__(self, other):
        return self.__add__(other)

    def __add__(self, other):
        if other == 0:
            return self
        match (self, other):
            case (
                Recurrence(base=e, op=operator.add, increment=f),
                int(g),
            ):
                return Recurrence(base=e + g, op=operator.add, increment=f)
            case (
                Recurrence(base=e, op=operator.add, increment=f),
                Recurrence(base=g, op=operator.add, increment=h),
            ):
                return Recurrence(base=e + g, op=operator.add, increment=f + h)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __mul__(self, other):
        if other == 1:
            return self
        elif other == 0:
            return 0
        match (self, other):
            case (
                Recurrence(base=e, op=operator.add, increment=f),
                int(g),
            ):
                return Recurrence(base=e * g, op=operator.add, increment=g * f)
            case (
                Recurrence(base=e, op=operator.mul, increment=f),
                int(g),
            ):
                return Recurrence(base=e * g, op=operator.mul, increment=f)
            case (
                Recurrence(base=e, op=operator.add, increment=f),
                Recurrence(base=g, op=operator.add, increment=h),
            ) as incs:
                return Recurrence(
                    base=e * g,
                    op=operator.add,
                    increment=incs[0] * h + incs[1] * f + f * h,
                )

    def normalize(self):
        match self:
            case Recurrence(base=0, op=operator.mul, increment=_):
                return 0
            case Recurrence(base=b, op=operator.add, increment=0) | Recurrence(
                base=b, op=operator.mul, increment=1
            ):
                return b
            case Recurrence(increment=Recurrence()) as r:
                return Recurrence(
                    base=r.base,
                    op=r.op,
                    increment=r.increment.normalize(),
                )
            case _:
                return self

    @staticmethod
    def from_assign(assign: Assign, induction_vars: dict[str, "Recurrence"]):
        match assign:
            case Increment(lhs=_, rhs=rhs):
                return Recurrence(
                    base=0,
                    op=operator.add,
                    increment=Recurrence.from_ast(rhs, induction_vars),
                )
            case Assign(lhs=_, rhs=_):
                # It's a cop out: I'm not supporting arbitrary assignments in
                # the loop body. Sorry, friendo.
                raise ValueError("Unsupported assign in loop body")

    @staticmethod
    def from_ast(tree: ast.AST, induction_vars: dict[str, "Recurrence"]):
        match tree:
            case ast.Module(body=[ast.Expr(value=value)]):
                # Inputs are single arithmetic expressions, so the
                # Module([Expr]) in which ast.parse wraps the expression can be
                # ignored.
                return Recurrence.from_ast(value, induction_vars=induction_vars)
            case ast.Constant(value=value):
                return value
            case ast.Name(id=name):
                if name in induction_vars:
                    return induction_vars[name]
                else:
                    raise ValueError(f"Missing SCEV for {name}")
            case ast.BinOp(left=left, op=op, right=right):
                parsed_left = Recurrence.from_ast(left, induction_vars=induction_vars)
                parsed_right = Recurrence.from_ast(right, induction_vars=induction_vars)
                sub = False
                match op:
                    case ast.Add():
                        parsed_op = operator.add
                    case ast.Sub():
                        parsed_op = operator.add
                        sub = True
                    case ast.Mult():
                        parsed_op = operator.mul
                if sub:
                    parsed_right = -1 * parsed_right
                return parsed_op(parsed_left, parsed_right).normalize()


@dataclass
class Loop:
    header: list[Assign]
    body: list[Assign]
    context: dict[str, Recurrence]


class UniqueId:
    def __init__(self):
        self.i = 0

    def get(self):
        output = self.i
        self.i += 1
        return output


def reduce_strength(loop: Loop) -> Loop:
    """A simple strength reduction for a loop."""
    body_scevs = {
        stmt.lhs: Recurrence.from_assign(stmt, induction_vars=loop.context)
        for stmt in loop.body
    }

    header: list[Assign] = []
    body: list[Assign] = []
    # assign variable names to each sub-recurrence
    vars: dict[str, Recurrence] = dict()
    ids: dict[Recurrence, str] = dict()
    var_id = UniqueId()

    def assign_var_names(recurrence):
        var_name = f"t{var_id.get()}"
        vars[var_name] = recurrence
        ids[recurrence] = var_name

    def make_header(rec):
        header.append(
            Assign(
                lhs=ast.Name(id=ids[rec]),
                rhs=ast.Constant(value=rec if isinstance(rec, int) else rec.base),
            ),
        )

    def make_body(recurrence):
        if isinstance(recurrence, int):
            return
        body.append(
            Increment(
                lhs=ast.Name(id=ids[recurrence]),
                rhs=ast.Name(id=ids[recurrence.increment]),
            ),
        )

    for lhs, scev in body_scevs.items():
        scev.visit(assign_var_names)
        scev.visit(make_header)
        scev.visit(make_body)
        body.append(Assign(lhs=lhs, rhs=ast.Name(id=ids[scev])))

    return Loop(header=header, body=body, context=dict())
