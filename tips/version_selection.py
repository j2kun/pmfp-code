from collections import deque
from dataclasses import dataclass
from pysat.solvers import Minisat22
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import Tuple


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


NameIndex = Callable[[Name, Version], PackageVersion]
IdIndex = Callable[[int], PackageVersion]


def select_dependent_versions(
    package: PackageVersion,
    package_index: NameIndex,
    id_index: IdIndex,
) -> Dict[Name, Version]:
    '''Select versions of dependent packages to install, or report impossible.'''
    solver = Minisat22()

    clauses = []
    to_process = deque([package])
    processed = set()

    while to_process:
        next_package = to_process.pop()
        processed.add(next_package)
        if not next_package.dependencies:
            continue

        '''
        For each dependency DEP, we need a clause that says 
        "if next_package is chosen, one of DEP's allowed versions 
        must be installed."

            next_package => (DEP_v1 OR DEP_v2 OR ...)

        This is equivalent to

            (!next_package OR DEP_v1 OR DEP_v2 OR ...)
        '''
        clause = [-next_package.package_id]
        for name, allowed_versions in next_package.dependencies:
            for version in allowed_versions:
                dep = package_index(name, version)
                clause.append(dep.package_id)
                if dep not in processed:
                    to_process.appendleft(dep)

        clauses.append(clause)

    if len(processed) == 1:
        return {package.name: package.version}
    
    for clause in clauses:
        solver.add_clause(clause)

    # solve with the assumption that the input package is chosen
    result = solver.solve(assumptions=[package.package_id])

    if not result:
        raise ValueError("Infeasible!")
        # return solver.get_core()

    return dict(
        id_index(abs(v)).as_tuple()
        for v in solver.get_model() if v > 0
    )
