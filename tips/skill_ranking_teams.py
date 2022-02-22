'''A TrueSkill-like team-based skill ranking algorithm.'''

from dataclasses import dataclass
from functools import reduce
import math
from typing import Dict
from typing import Iterable
from typing import List
from typing import NewType
from typing import Optional
from typing import Union

DEFAULT_STD_DEV = 25 / 3
DEFAULT_VARIANCE = DEFAULT_STD_DEV ** 2
DEFAULT_MEAN = 25

SKILL_CLASS_WIDTH = (DEFAULT_STD_DEV / 2) ** 2
ADDITIVE_DYNAMICS_FACTOR = (DEFAULT_STD_DEV / 100) ** 2
DRAW_PROBABILITY = 0.01  # 1 percent


@dataclass(frozen=True)
class Gaussian:
    '''A class representing a Gaussian distribution.

    Instead of using the mean and variance of the Gaussian, TrueSkill uses the
    precision and "precision-adjusted mean." These can be translated back to
    the mean and variance easily, and the benefit is that a product or division
    of two Gaussians corresponds to simpler arithmetic operations on precision
    and precision-adjusted mean than the same operations would require for mean
    and variance.
    '''
    precision: float = 1 / DEFAULT_VARIANCE
    precision_adjusted_mean: float = DEFAULT_MEAN / DEFAULT_VARIANCE

    @staticmethod
    def from_mean_variance(mean, variance):
        return Gaussian(precision=1 / variance,
                        precision_adjusted_mean=mean / variance)

    def __mul__(self, other: "Gaussian") -> "Gaussian":
        return Gaussian(
            precision=self.precision + other.precision,
            precision_adjusted_mean=(
                self.precision_adjusted_mean + other.precision_adjusted_mean))

    def __truediv__(self, other: "Gaussian") -> "Gaussian":
        return Gaussian(
            precision=self.precision - other.precision,
            precision_adjusted_mean=(
                self.precision_adjusted_mean - other.precision_adjusted_mean))

    def mean(self) -> float:
        return self.precision_adjusted_mean / self.precision

    def stddev(self) -> float:
        return math.sqrt(1 / self.precision)

    def __repr__(self):
        return f'N({self.mean():.2f}, {self.stddev():.2f})'


@dataclass(frozen=True)
class CumulativeGaussian:
    '''A class representing a cumulative Gaussian distribution.'''
    mean: float
    stddev: float

    def approx_mul(self, other: "Gaussian") -> "Gaussian":
        '''Approximate the product of a cumulative Gaussian and a Gaussian as a Gaussian.'''
        # TODO: finish
        return Gaussian.from_mean_variance(self.mean, self.stddev**2)

    def __repr__(self):
        return f'CumGauss(x - {self.mean:.2f}/ ({self.stddev:.2f} sqrt(2)))'


Message = Union[Gaussian, int]


def product(gs: Iterable[Gaussian]) -> Gaussian:
    return reduce(lambda g1, g2: g1 * g2, gs)


class Node:
    '''A node of the factor graph.'''

    def __init__(self):
        self.outgoing_messages: Dict[Node, Message] = {}

    def outgoing_message(self, destination: "Node", context: Optional[Gaussian] = None):
        if destination in self.outgoing_messages:
            msg = self.outgoing_messages[destination]
            print(f"(cached) Message from {self} to {destination}: {msg}")
            return msg
        print(f"Computing new message from {self} to {destination}...")
        msg = self.compute_outgoing_message(destination, context)
        self.outgoing_messages[destination] = msg
        print(f"Message from {self} to {destination}: {msg}")
        return msg

    def compute_outgoing_message(self, destination: "Node", context: Optional[Gaussian]) -> Message:
        raise NotImplementedError()


class VariableNode(Node):
    '''Nodes from one half of the bipartite factor graph, corresponding to variables.'''

    def __init__(self, name: str, observed_value: Optional[int] = None):
        super().__init__()
        self.name = name
        self.observed_value = observed_value
        self.factors: List[Node] = list()

    def add_factor(self, factor: Node) -> None:
        self.factors.append(factor)

    def compute_outgoing_message(
            self, destination: Node, context: Optional[Gaussian]) -> Message:
        if self.observed_value:
            return self.observed_value
        return product(
            factor.outgoing_message(self, context=None)
            for factor in self.factors if factor != destination
        )

    def marginal_distribution(self) -> Gaussian:
        return product(
            factor.outgoing_message(self, context=None)
            for factor in self.factors
        )

    def __repr__(self) -> str:
        return self.name


class PriorFactor(Node):
    '''A node in the factor graph for a prior on a variable.'''

    def __init__(self, variable: VariableNode, prior: Gaussian):
        super().__init__()
        self.variable = variable
        self.prior = prior

    def compute_outgoing_message(
            self, destination: Node, context: Optional[Gaussian]) -> Message:
        if destination != self.variable:
            raise ValueError(f"Prior only sends messages to {self.variable.name}")
        return self.prior

    def __repr__(self) -> str:
        return f"Prior_{self.prior}({self.variable.name})"


class GaussianFactor(Node):
    '''A node in the factor graph for a Gaussian with a variable mean.'''

    def __init__(self, mean_node: VariableNode, output_node: VariableNode, variance: float):
        super().__init__()
        self.mean_node = mean_node
        self.output_node = output_node
        self.variance = variance

    def compute_outgoing_message(
            self, destination: Node, context: Optional[Gaussian]) -> Message:
        del context  # unused
        other_node = self.output_node if destination == self.mean_node else self.mean_node
        msg = other_node.outgoing_message(self)
        return Gaussian.from_mean_variance(mean=msg.mean(), variance=msg.variance() + self.variance)

    def __repr__(self) -> str:
        return f"GaussianFactor(mean={self.mean_node.name})"


@dataclass
class ComparisonFactor(Node):
    '''A node in the factor graph for a left > right comparison.'''

    def __init__(self, left: VariableNode, right: VariableNode, output: VariableNode):
        super().__init__()
        self.left = left
        self.right = right
        self.output = output

    def compute_outgoing_message(
            self, destination: Node, context: Optional[Gaussian]) -> Message:
        integrate_over = self.left if destination == self.right else self.right
        message_to_integrate = integrate_over.outgoing_message(self)
        output_obs = self.output.outgoing_message(self)
        if output_obs > 0 and destination == self.left:
            message = CumulativeGaussian(
                mean=message_to_integrate.mean(),
                stddev=message_to_integrate.stddev(),
            )

        approximate_message = message
        return approximate_message

    def __repr__(self) -> str:
        return f"ComparisonFactor({self.output.name} = {self.left.name} > {self.right.name})"


def two_player_update(p1_prior: Gaussian, p2_prior: Gaussian, outcome: int):
    '''A two-player skill update without draws, via factor graphs.'''

    # construct the factor graph
    p1_skill = VariableNode("p1_skill")
    p2_skill = VariableNode("p2_skill")
    observed = VariableNode("outcome", observed_value=outcome)

    p1_skill_prior = PriorFactor(variable=p1_skill, prior=Gaussian())
    p2_skill_prior = PriorFactor(variable=p2_skill, prior=Gaussian())
    p1_skill.add_factor(p1_skill_prior)
    p2_skill.add_factor(p2_skill_prior)

    p1_perf = VariableNode("p1_perf")
    p2_perf = VariableNode("p1_perf")

    # the mean is set by a dependent variable, variance is constant
    p1_perf_factor = GaussianFactor(
        mean_node=p1_skill, output_node=p1_perf, variance=SKILL_CLASS_WIDTH)
    p2_perf_factor = GaussianFactor(
        mean_node=p2_skill, output_node=p2_perf, variance=SKILL_CLASS_WIDTH)

    p1_skill.add_factor(p1_perf_factor)
    p2_skill.add_factor(p2_perf_factor)
    p1_perf.add_factor(p1_perf_factor)
    p2_perf.add_factor(p2_perf_factor)

    comparison_factor = ComparisonFactor(left=p1_perf, right=p2_perf, output=observed)
    p1_perf.add_factor(comparison_factor)
    p2_perf.add_factor(comparison_factor)
    observed.add_factor(comparison_factor)

    p1_new_prior = p1_skill.marginal_distribution()
    p2_new_prior = p2_skill.marginal_distribution()

    return p1_new_prior, p2_new_prior


@dataclass
class Team:
    members: List[int]


PlayerRatings = NewType("PlayerRatings", Dict[int, Gaussian])


def update_ratings(team1: Team, team2: Team, ratings: PlayerRatings, outcome: int) -> PlayerRatings:
    pass
