from hypothesis import assume
from hypothesis import given
from hypothesis.strategies import booleans
from hypothesis.strategies import composite
from hypothesis.strategies import integers
from itertools import groupby
from itertools import takewhile
import pytest

from tips.version_selection import PackageVersion
from tips.version_selection import select_dependent_versions


class IdIndex:
    def __init__(self, packages):
        self.id_index = {p.package_id: p for p in packages}

    def __str__(self):
        return str(self.id_index)

    def __repr__(self):
        return str(self.id_index)

    def __call__(self, package_id):
        return self.id_index[package_id]


class NameVerIndex:
    def __init__(self, packages):
        self.name_ver_index = {
            key: {p.version: p for p in group}
            for (key, group) in groupby(packages, lambda p: p.name)
        }

    def __str__(self):
        return str(self.name_ver_index)

    def __repr__(self):
        return str(self.name_ver_index)

    def __call__(self, name):
        return self.name_ver_index[name]


def build_indices(packages):
    return (
        NameVerIndex(packages),
        IdIndex(packages),
    )


def test_no_deps():
    package = PackageVersion(
        package_id=1,
        name="foo",
        version=1,
        dependencies=tuple(),
    )
    chosen = select_dependent_versions(package, *build_indices([package]))

    assert chosen == {"foo": 1}


def test_single_dep():
    package = PackageVersion(
        package_id=1,
        name="foo",
        version=1,
        dependencies=(("bar", (1,)),),
    )
    dep = PackageVersion(
        package_id=2,
        name="bar",
        version=1,
        dependencies=tuple(),
    )
    chosen = select_dependent_versions(package, *build_indices([package, dep]))

    assert chosen == {"foo": 1, "bar": 1}


def test_branch_deps():
    p1 = PackageVersion(
        package_id=1,
        name="foo",
        version=1,
        dependencies=(
            ("bar", (1,)),
            ("baz", (1,)),
        ),
    )
    p2 = PackageVersion(
        package_id=2,
        name="bar",
        version=1,
        dependencies=tuple(),
    )
    p3 = PackageVersion(
        package_id=3,
        name="baz",
        version=1,
        dependencies=tuple(),
    )
    chosen = select_dependent_versions(p1, *build_indices([p1, p2, p3]))

    assert chosen == {"foo": 1, "bar": 1, "baz": 1}


def test_diamond_deps_feasible():
    p1 = PackageVersion(
        package_id=1,
        name="foo",
        version=1,
        dependencies=(
            ("bar", (1,)),
            ("baz", (1,)),
        ),
    )
    p2 = PackageVersion(
        package_id=2,
        name="bar",
        version=1,
        dependencies=(
            (
                "quux",
                (
                    1,
                    2,
                ),
            ),
        ),
    )
    p3 = PackageVersion(
        package_id=3,
        name="baz",
        version=1,
        dependencies=(("quux", (1,)),),
    )
    p4 = PackageVersion(
        package_id=4,
        name="quux",
        version=1,
        dependencies=tuple(),
    )
    p5 = PackageVersion(
        package_id=5,
        name="quux",
        version=2,
        dependencies=tuple(),
    )
    chosen = select_dependent_versions(p1, *build_indices([p1, p2, p3, p4, p5]))

    assert chosen == {"foo": 1, "bar": 1, "baz": 1, "quux": 1}


def test_diamond_deps_infeasible():
    p1 = PackageVersion(
        package_id=1,
        name="foo",
        version=1,
        dependencies=(
            ("bar", (1,)),
            ("baz", (1,)),
        ),
    )
    p2 = PackageVersion(
        package_id=2,
        name="bar",
        version=1,
        dependencies=(("quux", (1,)),),
    )
    p3 = PackageVersion(
        package_id=3,
        name="baz",
        version=1,
        dependencies=(("quux", (2,)),),
    )
    p4 = PackageVersion(
        package_id=4,
        name="quux",
        version=1,
        dependencies=tuple(),
    )
    p5 = PackageVersion(
        package_id=5,
        name="quux",
        version=2,
        dependencies=tuple(),
    )

    with pytest.raises(ValueError):
        select_dependent_versions(p1, *build_indices([p1, p2, p3, p4, p5]))


@composite
def dependency_graph(
    draw,
    dependency_decider=booleans(),
    num_packages=integers(min_value=1, max_value=10),
    num_versions=integers(min_value=1, max_value=10),
):
    names = [f"package_{i}" for i in range(draw(num_packages))]
    packages = []
    i = 1
    for name in names:
        previous_names = list(takewhile(lambda n: n != name, names))
        index = NameVerIndex(packages)
        for version in range(1, 1 + draw(num_versions)):
            dependencies = []
            for dep_name in previous_names:
                if draw(dependency_decider):
                    legal_dep_versions = []
                    for dep_version in list(index(dep_name).keys()):
                        if draw(dependency_decider):
                            legal_dep_versions.append(dep_version)
                    if legal_dep_versions:
                        dependencies.append((dep_name, tuple(legal_dep_versions)))

            packages.append(
                PackageVersion(
                    package_id=i,
                    name=name,
                    version=version,
                    dependencies=tuple(dependencies),
                ),
            )
            i += 1

    return packages


@given(dependency_graph())
def test_arbitrary_dep_graph_feasible(graph):
    try:
        chosen = select_dependent_versions(graph[-1], *build_indices(graph))
    except Exception:
        chosen = dict()

    # either infeasible or no deps on the last package
    assume(chosen != dict())

    index = NameVerIndex(graph)
    for (name, version) in chosen.items():
        p = index(name)[version]
        for dep_name, allowed_versions in p.dependencies:
            assert dep_name in chosen
            assert chosen[dep_name] in allowed_versions


@given(dependency_graph())
def test_arbitrary_dep_graph_infeasible(graph):
    try:
        select_dependent_versions(graph[-1], *build_indices(graph))
        assume(False)
    except Exception:
        pass
