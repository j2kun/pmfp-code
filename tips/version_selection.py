from collections import deque
from dataclasses import dataclass
from itertools import combinations
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import Tuple

from pysat.solvers import Glucose4

Name = str
Version = int


@dataclass(frozen=True)
class PackageVersion:
    package_id: int  # unique positive int, for indexing variables
    name: Name
    version: Version
    dependencies: Iterable[Tuple[Name, Iterable[Version]]]

    def as_tuple(self):
        return (self.name, self.version)


NameIndex = Callable[[Name], Dict[Version, PackageVersion]]
IdIndex = Callable[[int], PackageVersion]


def select_dependent_versions(
    package: PackageVersion,
    package_index: NameIndex,
    id_index: IdIndex,
) -> Dict[Name, Version]:
    """Select versions of dependent packages to install, or report impossible."""
    solver = Glucose4()

    clauses = []
    to_process = deque([package])
    processed = set()

    while to_process:
        next_package = to_process.pop()
        processed.add(next_package)
        """At most one version of any package may be installed.  While SAT solvers admit
        many ways to model "cardinality constraints", we choose a simple "pairwise"
        model that adds the constraint.

        v1 => NOT v2

        for two distinct versions v1, v2, of the same package. This is equivalent to

        (NOT v1) OR (NOT v2)
        """
        versions = [p.package_id for p in package_index(next_package.name).values()]
        for v1, v2 in combinations(versions, 2):
            clauses.append([-v1, -v2])

        """
        For each dependency DEP, we need a clause that says
        "if next_package is chosen, one of DEP's allowed versions
        must be installed."

            next_package => (DEP_v1 OR DEP_v2 OR ...)

        This is equivalent to

            (!next_package OR DEP_v1 OR DEP_v2 OR ...)
        """
        for name, allowed_versions in next_package.dependencies:
            clause = [-next_package.package_id]
            for version in allowed_versions:
                dep = package_index(name)[version]
                clause.append(dep.package_id)
                if dep not in processed:
                    to_process.appendleft(dep)
            clauses.append(clause)

    for clause in clauses:
        solver.add_clause(clause)

    # solve with the assumption that the input package is chosen
    if not solver.solve(assumptions=[package.package_id]):
        # print(clauses)
        # return solver.get_core()
        raise ValueError("Infeasible!")

    return dict(id_index(v).as_tuple() for v in solver.get_model() if v > 0)
