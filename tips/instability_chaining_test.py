from hypothesis import given
from hypothesis.strategies import composite
from hypothesis.strategies import integers
from hypothesis.strategies import permutations

from instability_chaining import Couple
from instability_chaining import Matching
from instability_chaining import ResidencyProgram
from instability_chaining import Student
from instability_chaining import stable_matching


@composite
def market(
    draw,
    num_students=integers(min_value=1, max_value=5),
    num_programs=integers(min_value=1, max_value=5),
    program_capacity=integers(min_value=1, max_value=5),
):
    student_ids = list(range(draw(num_students)))
    program_ids = list(range(draw(num_programs)))

    students = []
    programs = []

    for student_id in student_ids:
        preferences = draw(permutations(program_ids))
        students.append(
            Student(
                id=student_id,
                preferences=preferences,
                best_unrejected=0,
            )
        )

    for program_id in program_ids:
        preferences = draw(permutations(student_ids))
        programs.append(
            ResidencyResidencyProgram(
                id=program_id,
                preferences=preferences,
                capacity=draw(program_capacity),
            )
        )

    # TODO: make some students couples!

    return (students, programs)


@given(market())
def test_stability(students_and_programs):
    students, programs = students_and_programs
    matching = stable_matching(students, programs)
    unstable_pair = find_unstable_pair(students, programs, matching)

    def err_msg():
        return (
            f"\n{unstable_pair[0]},"
            f"\n{unstable_pair[1]},"
            f"\nform an unstable pair in matching \n{matching}"
        )

    assert unstable_pair is None, err_msg()


"""
TODO: test cases

 - Test finding unstable pairs in isolation
 - Test a couple-free iteration
 - Test with one couple
 - Test with a chain of two couples bumping each other
 - Test with a cycle

TODO: Cycle detection & randomization

TODO: re-read algorithm paper https://web.stanford.edu/~alroth/papers/rothperansonaer.PDF
to make sure I'm not missing anything.
"""


def test_find_unstable_pair_stable():
    students = [Student(id=0, preferences=[0, 1]), Student(id=1, preferences=[1, 0])]
    programs = [
        ResidencyProgram(id=0, capacity=1, preferences=[0, 1]),
        ResidencyProgram(id=1, capacity=1, preferences=[1, 0]),
    ]
    matching = Matching(matches={students[0]: programs[0], students[1]: programs[1]})
    assert find_unstable_pair(students, programs, matching) is None


def test_find_unstable_pair_unstable():
    students = [Student(id=0, preferences=[1, 0]), Student(id=1, preferences=[1, 0])]
    programs = [
        ResidencyProgram(id=0, capacity=1, preferences=[0, 1]),
        ResidencyProgram(id=1, capacity=1, preferences=[0, 1]),
    ]
    matching = Matching(matches={students[0]: programs[0], students[1]: programs[1]})
    result = find_unstable_pair(students, programs, matching)
    assert result == (students[0], programs[1])


def test_stable_matching_two():
    students = [Student(id=0, preferences=[0, 1]), Student(id=1, preferences=[1, 0])]
    programs = [
        ResidencyProgram(id=0, capacity=1, preferences=[0, 1]),
        ResidencyProgram(id=1, capacity=1, preferences=[1, 0]),
    ]
    matching = stable_matching(students, programs)
    assert matching.matches == {students[0]: programs[0], students[1]: programs[1]}
    assert find_unstable_pair(students, programs, matching) is None


def test_stable_matching_six():
    students = [
        Student(id=0, preferences=[3, 5, 4, 2, 1, 0]),
        Student(id=1, preferences=[2, 3, 1, 0, 4, 5]),
        Student(id=2, preferences=[5, 2, 1, 0, 3, 4]),
        Student(id=3, preferences=[0, 1, 2, 3, 4, 5]),
        Student(id=4, preferences=[4, 5, 1, 2, 0, 3]),
        Student(id=5, preferences=[0, 1, 2, 3, 4, 5]),
    ]

    programs = [
        ResidencyProgram(id=0, capacity=1, preferences=[3, 5, 4, 2, 1, 0]),
        ResidencyProgram(id=1, capacity=1, preferences=[2, 3, 1, 0, 4, 5]),
        ResidencyProgram(id=2, capacity=1, preferences=[5, 2, 1, 0, 3, 4]),
        ResidencyProgram(id=3, capacity=1, preferences=[0, 1, 2, 3, 4, 5]),
        ResidencyProgram(id=4, capacity=1, preferences=[4, 5, 1, 2, 0, 3]),
        ResidencyProgram(id=5, capacity=1, preferences=[0, 1, 2, 3, 4, 5]),
    ]

    matching = stable_matching(students, programs)
    assert matching.matches == {
        students[0]: programs[3],
        students[1]: programs[2],
        students[2]: programs[5],
        students[3]: programs[0],
        students[4]: programs[4],
        students[5]: programs[1],
    }
    assert find_unstable_pair(students, programs, matching) is None


def test_stable_matching_all_tied():
    students = [
        Student(id=0, preferences=[5, 4, 3, 2, 1, 0]),
        Student(id=1, preferences=[5, 4, 3, 2, 1, 0]),
        Student(id=2, preferences=[5, 4, 3, 2, 1, 0]),
        Student(id=3, preferences=[5, 4, 3, 2, 1, 0]),
        Student(id=4, preferences=[5, 4, 3, 2, 1, 0]),
        Student(id=5, preferences=[5, 4, 3, 2, 1, 0]),
    ]

    programs = [
        ResidencyProgram(id=0, capacity=1, preferences=[0, 1, 2, 3, 4, 5]),
        ResidencyProgram(id=1, capacity=1, preferences=[0, 1, 2, 3, 4, 5]),
        ResidencyProgram(id=2, capacity=1, preferences=[0, 1, 2, 3, 4, 5]),
        ResidencyProgram(id=3, capacity=1, preferences=[0, 1, 2, 3, 4, 5]),
        ResidencyProgram(id=4, capacity=1, preferences=[0, 1, 2, 3, 4, 5]),
        ResidencyProgram(id=5, capacity=1, preferences=[0, 1, 2, 3, 4, 5]),
    ]

    matching = stable_matching(students, programs)
    assert matching.matches == {
        students[0]: programs[5],
        students[1]: programs[4],
        students[2]: programs[3],
        students[3]: programs[2],
        students[4]: programs[1],
        students[5]: programs[0],
    }
    assert find_unstable_pair(students, programs, matching) is None


def test_stable_matching_capacity_2():
    students = [
        Student(id=0, preferences=[5, 4, 3, 2, 1, 0]),
        Student(id=1, preferences=[5, 4, 3, 2, 1, 0]),
        Student(id=2, preferences=[5, 4, 3, 2, 1, 0]),
        Student(id=3, preferences=[5, 4, 3, 2, 1, 0]),
        Student(id=4, preferences=[5, 4, 3, 2, 1, 0]),
        Student(id=5, preferences=[5, 4, 3, 2, 1, 0]),
    ]

    programs = [
        ResidencyProgram(id=0, capacity=2, preferences=[0, 1, 2, 3, 4, 5]),
        ResidencyProgram(id=1, capacity=2, preferences=[0, 1, 2, 3, 4, 5]),
        ResidencyProgram(id=2, capacity=2, preferences=[0, 1, 2, 3, 4, 5]),
        ResidencyProgram(id=3, capacity=2, preferences=[0, 1, 2, 3, 4, 5]),
        ResidencyProgram(id=4, capacity=2, preferences=[0, 1, 2, 3, 4, 5]),
        ResidencyProgram(id=5, capacity=2, preferences=[0, 1, 2, 3, 4, 5]),
    ]

    matching = stable_matching(students, programs)
    assert matching.matches == {
        students[0]: programs[5],
        students[1]: programs[5],
        students[2]: programs[4],
        students[3]: programs[4],
        students[4]: programs[3],
        students[5]: programs[3],
    }
    assert find_unstable_pair(students, programs, matching) is None


def test_stable_matching_limited_student_preference_lists():
    students = [
        Student(id=0, preferences=[5, 4]),
        Student(id=1, preferences=[5, 4]),
        Student(id=2, preferences=[5, 4]),
        Student(id=3, preferences=[5, 4]),
        Student(id=4, preferences=[5, 4]),
        Student(id=5, preferences=[5, 4]),
    ]

    programs = [
        ResidencyProgram(id=0, capacity=2, preferences=[0, 1, 2, 3, 4, 5]),
        ResidencyProgram(id=1, capacity=2, preferences=[0, 1, 2, 3, 4, 5]),
        ResidencyProgram(id=2, capacity=2, preferences=[0, 1, 2, 3, 4, 5]),
        ResidencyProgram(id=3, capacity=2, preferences=[0, 1, 2, 3, 4, 5]),
        ResidencyProgram(id=4, capacity=2, preferences=[0, 1, 2, 3, 4, 5]),
        ResidencyProgram(id=5, capacity=2, preferences=[0, 1, 2, 3, 4, 5]),
    ]

    matching = stable_matching(students, programs)
    assert matching.matches == {
        students[0]: programs[5],
        students[1]: programs[5],
        students[2]: programs[4],
        students[3]: programs[4],
    }
    assert matching.unassigned == {
        students[4],
        students[5],
    }
    assert find_unstable_pair(students, programs, matching) is None
