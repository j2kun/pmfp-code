from dataclasses import dataclass
from typing import Callable
from typing import Union
import ast
import operator

# only operator.add and operator.mul are supported
Operator = Callable[[int, int], int]
allowed_ops = [operator.add, operator.mul]


class ExprMeta(type):
    def __instancecheck__(self, instance):
        is_int = isinstance(instance, int)
        is_recurrence = isinstance(instance, Recurrence)
        return is_int or (
            not is_recurrence
            and hasattr(instance, "__add__")
            and hasattr(instance, "__mul__")
        )


class Expr(metaclass=ExprMeta):
    def __str__(self):
        return repr(self)

    def __add__(self, other):
        return Add(self, other)

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        return Mul(self, other)

    def __rmul__(self, other):
        return self.__mul__(other)


class Ref(Expr):
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return self.name


@dataclass
class Add(Expr):
    left: Union[int, Expr]
    right: Union[int, Expr]

    def __repr__(self):
        return f"{repr(self.left)} + {repr(self.right)}"


@dataclass
class Mul(Expr):
    left: Union[int, Expr]
    right: Union[int, Expr]

    def __repr__(self):
        return f"{repr(self.left)} * {repr(self.right)}"


class Recurrence:
    # TODO: really, base should be a value or a reference
    def __init__(
        self,
        base: Union[int, Expr],
        op: Operator,
        increment: Union["Recurrence", int, Expr],
    ):
        assert op in allowed_ops
        assert isinstance(base, int) or isinstance(base, Expr)
        self.base = base
        self.op = op
        self.increment = increment

    @staticmethod
    def constant(x: Union[int, Expr]):
        return Recurrence(base=x, op=operator.add, increment=0)

    def evaluate(self, i):
        if i == 0:
            return self.base

        if isinstance(self.increment, int):
            increment = self.increment
        else:
            increment = self.increment.evaluate(i - 1)

        return self.op(self.evaluate(i - 1), increment)

    def br_notation(self, flatten=True):
        match self.op:
            case operator.add:
                op_str = "+"
            case operator.mul:
                op_str = "*"

        nested = repr(self.increment)
        if flatten:
            nested = nested.strip("{").strip("}")
        return f"{{{self.base}, {op_str}, {nested}}}"

    def __repr__(self):
        return self.br_notation(flatten=True)

    def __eq__(self, other):
        return isinstance(other, Recurrence) and repr(self) == repr(other)

    def __radd__(self, other):
        return self.__add__(other)

    def __add__(self, other):
        if self.op == operator.add and other == 0:
            return self
        elif self.op == operator.mul and other == 1:
            return self
        elif self.op == operator.mul and other == 0:
            return 0
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
            case _:
                raise ValueError(f"Unsupported add {self} + {other}")

    def __rmul__(self, other):
        return self.__mul__(other)

    def __mul__(self, other):
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
            ):
                inc1 = Recurrence(base=e, op=operator.add, increment=f)
                inc2 = Recurrence(base=g, op=operator.add, increment=h)
                return Recurrence(
                    base=e * g,
                    op=operator.add,
                    increment=inc1 * h + inc2 * f + f * h,
                )
            case _:
                raise ValueError(f"Unsupported mul {self} * {other}")

    def normalize(self):
        match self:
            case (
                Recurrence(base=b, op=operator.add, increment=0)
                | Recurrence(base=b, op=operator.mul, increment=1)
            ):
                return b
            case Recurrence(base=0, op=operator.mul, increment=_):
                return 0
            case Recurrence(increment=Recurrence()) as r:
                return Recurrence(
                    base=r.base, op=r.op, increment=r.increment.normalize()
                )
            case _:
                return self

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
                    return Recurrence.constant(Ref(name=name))
            case ast.BinOp(left=left, op=op, right=right):
                print("")
                print(f"Parsing {op} left={ast.dump(left, annotate_fields=False)}")
                parsed_left = Recurrence.from_ast(left, induction_vars=induction_vars)
                print(f"Parsing {op} right={ast.dump(right, annotate_fields=False)}")
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
                    case _:
                        raise ValueError(f"Unsupported op {op}")
                if sub:
                    parsed_right = -1 * parsed_right
                return parsed_op(parsed_left, parsed_right).normalize()
