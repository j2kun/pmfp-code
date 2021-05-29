from collections import defaultdict
from version_selection import PackageVersion
from version_selection import select_dependent_versions


class IdIndex:
    def __init__(self, packages):
        self.id_index = dict((p.package_id, p) for p in packages)

    def __str__(self):
        return str(self.id_index)

    def __repr__(self):
        return str(self.id_index)

    def __call__(self, package_id):
        return self.id_index.get(package_id)


class NameVerIndex:
    def __init__(self, packages):
        self.name_ver_index = dict(((p.name, p.version), p) for p in packages)

    def __str__(self):
        return str(self.name_ver_index)

    def __repr__(self):
        return str(self.name_ver_index)

    def __call__(self, name, ver):
        return self.name_ver_index.get((name, ver))


def build_indices(packages):
    return (
        NameVerIndex(packages),
        IdIndex(packages),
    )


def test_no_deps():
    package = PackageVersion(
        package_id=1,
        name='foo',
        version=1,
        dependencies=tuple(),
    )
    chosen = select_dependent_versions(
        package,
        *build_indices([package])
    )

    assert chosen == {'foo': 1}


def test_single_dep():
    package = PackageVersion(
        package_id=1,
        name='foo',
        version=1,
        dependencies=(
            ('bar', (1,)),
        ),
    )
    dep = PackageVersion(
        package_id=2,
        name='bar',
        version=1,
        dependencies=tuple(),
    )
    chosen = select_dependent_versions(
        package,
        *build_indices([package, dep])
    )

    assert chosen == {'foo': 1, 'bar': 1}
