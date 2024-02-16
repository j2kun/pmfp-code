"""Partition a graph into communities using Louvain with the lambdaCC objective."""
import itertools
import random
from dataclasses import dataclass

import igraph


@dataclass
class Community:
    id: int
    vertices: set[int]

    # total_vertex_weight is unused, see note below about the incorrect formula
    # from Shi et al.
    total_vertex_weight: float

    def __hash__(self):
        return hash(self.id)


def move_vertex(
    graph: igraph.Graph,
    vertex_id: int,
    current_community: Community,
    target_community: Community,
):
    """Move a vertex from one community to another, maintaining data structures."""
    current_community.vertices.remove(vertex_id)
    current_community.total_vertex_weight -= graph.vs[vertex_id]["weight"]
    target_community.vertices.add(vertex_id)
    target_community.total_vertex_weight += graph.vs[vertex_id]["weight"]
    graph.vs[vertex_id]["community"] = target_community.id


# We never need to compute the total objective, since the algorithm can always
# work with the incremental objective change, but I added it here for clarity.
def _compute_objective(graph: igraph.Graph, resolution: float) -> float:
    vertex_count = graph.vcount()
    objective = 0
    for i, j in itertools.combinations(range(vertex_count), 2):
        rescaled_weight = -resolution * graph.vs[i]["weight"] * graph.vs[j]["weight"]
        if graph.are_connected(i, j):
            rescaled_weight += graph.es[graph.get_eid(i, j)]["weight"]
        distance = 0 if graph.vs[i]["community"] == graph.vs[j]["community"] else 1
        objective += rescaled_weight * (1 - distance)

    return objective


def compute_objective_change(
    graph: igraph.Graph,
    resolution: float,
    vertex_id: int,
    current_community: Community,
    target_community: Community,
) -> float:
    """Compute the incremental change in the objective if vertex_id is moved to
    target_community."""
    if current_community.id == target_community.id:
        return 0
    vertex_weight = graph.vs[vertex_id]["weight"]
    """In Shi et al. (https://arxiv.org/abs/2108.01731), Appendix A claims that
    the commented out code is sufficient to compute the change in objective,
    which "depends solely on [the total vertex weights of the two clusters],
    and the weights of the edges from v to its neighbors in [the two clusters]."

    This is easily seen to be false: given two singleton clusters {u} and {v},
    where {v} is not adjacent to {u}, the change in objective from moving v to
    {u} is -resolution, but this formula would result in the change being zero.
    Instead, we compute the more inefficient (but correct) change by iterating
    over all vertices in the two clusters and computing the change for the
    edge/non-edge with v.

    Moreover, if we use the code below many of the test cases will fail because
    Louvain will enter an infinite loop, as the optimal vertex movement computed
    by this rule results in an occasional decrease in the objective function,
    violating monotonicity and causing the algorithm to cycle.

    target_term = 0
    current_term = 0
    vertex_weight = graph.vs[vertex_id]["weight"]
    for neighbor_id in graph.neighbors(vertex_id):
        edge_weight = graph.es[graph.get_eid(vertex_id, neighbor_id)]["weight"]
        neighbor_community_id = graph.vs[neighbor_id]["community"]
        if neighbor_community_id == current_community.id:
            current_term += edge_weight
            current_term -= (
                    resolution * vertex_weight * current_community.total_vertex_weight)
            current_term += resolution * vertex_weight * vertex_weight
        elif neighbor_community_id == target_community.id:
            target_term += edge_weight
            target_term -= (
                    resolution * vertex_weight * target_community.total_vertex_weight)
    change = target_term - current_term
    """

    change = 0
    for other_id in current_community.vertices | target_community.vertices:
        if other_id == vertex_id:
            continue
        other_community = graph.vs[other_id]["community"]
        rescaled_weight = -resolution * vertex_weight * graph.vs[other_id]["weight"]
        if graph.are_connected(vertex_id, other_id):
            rescaled_weight += graph.es[graph.get_eid(vertex_id, other_id)]["weight"]

        # This is more complicated than it needs to be, to match the formulas
        # in the paper https://arxiv.org/abs/2108.01731
        before_distance = 0 if other_community == current_community.id else 1
        after_distance = 0 if other_community == target_community.id else 1
        before_term = rescaled_weight * (1 - before_distance)
        after_term = rescaled_weight * (1 - after_distance)
        change += after_term - before_term

    return change


def louvain(graph: igraph.Graph, resolution: float, debug=False) -> set[Community]:
    """Partition the graph into communities using the sequential Louvain method.

    Args:
        graph: An igraph.Graph object. The graph's vertices and edges may have
          "weight" attributes specifying the vertex/edge weight, otherwise the
          weight is assumed to be 1.
        resolution: The lambda parameter for the lambdaCC objective
          function, between 0 and 1 exclusive.

    Returns:
        A set of sets of integers, where each set represents a community.
    """
    if "weight" not in graph.vs.attributes():
        graph.vs["weight"] = 1.0
    if "weight" not in graph.es.attributes():
        graph.es["weight"] = 1.0

    vertex_count = graph.vcount()
    vertices = list(range(vertex_count))
    graph.vs["community"] = vertices
    communities = {
        v["community"]: Community(v.index, {v.index}, v["weight"]) for v in graph.vs
    }

    if debug:
        print(f"Initial objective: {_compute_objective(graph, resolution)}")
    iterations = 0

    while True:
        if debug:
            print("\nCommunities at start of loop")
            for c in communities.values():
                print(f"\t{c}")
            print("")
        had_change = False
        for i in random.sample(range(vertex_count), vertex_count):
            current_community = communities[graph.vs[i]["community"]]
            objective_changes = [
                (
                    target_community,
                    compute_objective_change(
                        graph,
                        resolution,
                        i,
                        current_community,
                        target_community,
                    ),
                )
                for target_community in communities.values()
            ]
            best_community, largest_change = max(objective_changes, key=lambda x: x[1])
            if largest_change > 1e-08:
                if debug:
                    before_objective = _compute_objective(graph, resolution)
                    print(f"Objective before move={before_objective}")
                move_vertex(graph, i, current_community, best_community)
                if debug:
                    after_objective = _compute_objective(graph, resolution)
                    print(f"Objective after move={after_objective}")
                    if (
                        abs(abs(before_objective - after_objective) - largest_change)
                        > 1e-08
                    ):
                        raise ValueError(
                            f"Objective change does not match: {largest_change=:G} "
                            f"{before_objective=:G} {after_objective=:G}",
                        )
                had_change = True
                if debug:
                    print(f"{largest_change=}")
                if len(current_community.vertices) == 0:
                    del communities[current_community.id]

        iterations += 1
        if not had_change:
            break

    if debug:
        print(f"Final objective: {_compute_objective(graph, resolution)}")
    print(f"Total iterations: {iterations}")
    return set(communities.values())
