from itertools import combinations
import random

from hypothesis import given
from hypothesis.strategies import booleans
from hypothesis.strategies import composite
from hypothesis.strategies import integers
import pytest

from tips.topological_sort import topological_sort


def assert_satisfies_dependency_order(dag, sorted_nodes):
    # Test that a sorted output satisfies the dependency requirements of the
    # DAG.
    for node, dependents in dag.items():
        for dependent in dependents:
            node_index = sorted_nodes.index(node)
            dependent_index = sorted_nodes.index(dependent)
            assert node_index < dependent_index, sorted_nodes


@pytest.mark.parametrize(
    "dag,expected",
    [
        (dict(), []),
        ({1: []}, [1]),
        ({1: [2], 2: []}, [1, 2]),
        ({2: [1, 3], 1: [], 3: [1]}, [2, 3, 1]),
        ({0: [1], 1: [2], 2: [3], 3: [4], 4: []}, [0, 1, 2, 3, 4]),
    ],
)
def test_unique_toposort(dag, expected):
    assert expected == topological_sort(dag)


@pytest.mark.parametrize(
    "dag",
    [
        {
            1: [2, 3],
            2: [],
            3: [],
        },
        {
            1: [2, 3],
            2: [4],
            3: [4],
            4: [],
        },
        {
            1: [2, 4],
            2: [3, 5],
            3: [4, 6],
            4: [5],
            5: [],
            6: [],
        },
    ],
)
def test_non_unique_toposort(dag):
    assert_satisfies_dependency_order(dag, topological_sort(dag))


@pytest.mark.parametrize(
    "dag",
    [
        {
            1: [2],
        },
        {
            1: [2],
            2: [3],
            3: [4],
        },
    ],
)
def test_missing_keys(dag):
    with pytest.raises(KeyError):
        topological_sort(dag)


@pytest.mark.parametrize(
    "dag",
    [
        {
            1: [2],
            2: [1],
        },
        {
            1: [2],
            2: [3],
            3: [1],
        },
    ],
)
def test_loop_detection(dag):
    with pytest.raises(ValueError):
        topological_sort(dag)


@composite
def random_dag(
    draw,
    dependency_decider=booleans(),
    num_nodes=integers(min_value=1, max_value=100),
):
    """Generate a random DAG."""
    nodes = list(range(draw(num_nodes)))
    dag = {i: [] for i in nodes}
    random.shuffle(nodes)

    for i, j in combinations(nodes, 2):
        if draw(dependency_decider):
            dag[i].append(j)

    return dag


@given(random_dag())
def test_random_dag(dag):
    assert_satisfies_dependency_order(dag, topological_sort(dag))


@composite
def random_dag_with_loop(
    draw,
    dag_builder=random_dag(num_nodes=integers(min_value=1, max_value=50)),
    loop_member_picker=booleans(),
):
    """Generate a random DAG, then pick a random subset of nodes and form a
    directed loop.
    """
    dag = draw(dag_builder)

    # the cycle must have at least one node, and the picker may always return
    # false.
    chosen_nodes = {random.choice(list(dag.keys()))}

    for node in dag.keys():
        if draw(loop_member_picker):
            chosen_nodes.add(node)

    loop = list(chosen_nodes)
    random.shuffle(loop)

    for i, j in zip(loop, loop[1:]):
        dependents = set(dag[i])
        dependents.add(j)
        dag[i] = list(dependents)

    # add the last edge looping back to the first
    dependents = set(dag[loop[-1]])
    dependents.add(loop[0])
    dag[loop[-1]] = list(dependents)

    return dag


@given(random_dag_with_loop())
def test_err_on_random_dag_with_cycle(dag):
    with pytest.raises(ValueError):
        topological_sort(dag)
