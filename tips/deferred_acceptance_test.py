import hypothesis
from hypothesis import given
from hypothesis.strategies import composite
from hypothesis.strategies import integers
from hypothesis.strategies import permutations

from deferred_acceptance import Matching
from deferred_acceptance import School
from deferred_acceptance import Student
from deferred_acceptance import deferred_acceptance
from deferred_acceptance import find_unstable_pair


@composite
def market(
    draw,
    num_students=integers(min_value=1, max_value=5),
    num_schools=integers(min_value=1, max_value=5),
    school_capacity=integers(min_value=1, max_value=5),
):
    student_ids = list(range(draw(num_students)))
    school_ids = list(range(draw(num_schools)))

    students = []
    schools = []

    for student_id in student_ids:
        preferences = draw(permutations(school_ids))
        truncate_pt = draw(integers(min_value=1, max_value=len(school_ids)))
        students.append(
            Student(
                id=student_id,
                preferences=preferences[:truncate_pt],
                best_unrejected=0,
            )
        )

    for school_id in school_ids:
        preferences = draw(permutations(student_ids))
        schools.append(
            School(
                id=school_id,
                preferences=preferences,
                capacity=draw(school_capacity),
                held=list(),
            )
        )

    return (students, schools)


@given(market())
def test_stability(students_and_schools):
    students, schools = students_and_schools
    matching = deferred_acceptance(students, schools)
    unstable_pair = find_unstable_pair(students, schools, matching)

    def err_msg():
        return (
            f"\n{unstable_pair[0]},"
            f"\n{unstable_pair[1]},"
            f"\nform an unstable pair in matching \n{matching}"
        )

    assert unstable_pair is None, err_msg()


def test_find_unstable_pair_stable():
    students = [
        Student(id=0, preferences=[0, 1]),
        Student(id=1, preferences=[1, 0])
    ]
    schools = [
        School(id=0, capacity=1, preferences=[0, 1]),
        School(id=1, capacity=1, preferences=[1, 0])
    ]
    matching = Matching(
        matches={
            students[0]: schools[0],
            students[1]: schools[1]
        }
    )
    assert find_unstable_pair(students, schools, matching) is None


def test_find_unstable_pair_unstable():
    students = [
        Student(id=0, preferences=[1, 0]),
        Student(id=1, preferences=[1, 0])
    ]
    schools = [
        School(id=0, capacity=1, preferences=[0, 1]),
        School(id=1, capacity=1, preferences=[0, 1])
    ]
    matching = Matching(
        matches={
            students[0]: schools[0],
            students[1]: schools[1]
        }
    )
    result = find_unstable_pair(students, schools, matching)
    assert result == (students[0], schools[1])


def test_deferred_acceptance_two():
    students = [
        Student(id=0, preferences=[0, 1]),
        Student(id=1, preferences=[1, 0])
    ]
    schools = [
        School(id=0, capacity=1, preferences=[0, 1]),
        School(id=1, capacity=1, preferences=[1, 0])
    ]
    matching = deferred_acceptance(students, schools)
    assert matching.matches == {
        students[0]: schools[0],
        students[1]: schools[1]
    }
    assert find_unstable_pair(students, schools, matching) is None


def test_deferred_acceptance_six():
    students = [
        Student(id=0, preferences=[3, 5, 4, 2, 1, 0]),
        Student(id=1, preferences=[2, 3, 1, 0, 4, 5]),
        Student(id=2, preferences=[5, 2, 1, 0, 3, 4]),
        Student(id=3, preferences=[0, 1, 2, 3, 4, 5]),
        Student(id=4, preferences=[4, 5, 1, 2, 0, 3]),
        Student(id=5, preferences=[0, 1, 2, 3, 4, 5]),
    ]

    schools = [
        School(id=0, capacity=1, preferences=[3, 5, 4, 2, 1, 0]),
        School(id=1, capacity=1, preferences=[2, 3, 1, 0, 4, 5]),
        School(id=2, capacity=1, preferences=[5, 2, 1, 0, 3, 4]),
        School(id=3, capacity=1, preferences=[0, 1, 2, 3, 4, 5]),
        School(id=4, capacity=1, preferences=[4, 5, 1, 2, 0, 3]),
        School(id=5, capacity=1, preferences=[0, 1, 2, 3, 4, 5]),
    ]

    matching = deferred_acceptance(students, schools)
    assert matching.matches == {
        students[0]: schools[3],
        students[1]: schools[2],
        students[2]: schools[5],
        students[3]: schools[0],
        students[4]: schools[4],
        students[5]: schools[1],
    }
    assert find_unstable_pair(students, schools, matching) is None


def test_deferred_acceptance_all_tied():
    students = [
        Student(id=0, preferences=[5, 4, 3, 2, 1, 0]),
        Student(id=1, preferences=[5, 4, 3, 2, 1, 0]),
        Student(id=2, preferences=[5, 4, 3, 2, 1, 0]),
        Student(id=3, preferences=[5, 4, 3, 2, 1, 0]),
        Student(id=4, preferences=[5, 4, 3, 2, 1, 0]),
        Student(id=5, preferences=[5, 4, 3, 2, 1, 0]),
    ]

    schools = [
        School(id=0, capacity=1, preferences=[0, 1, 2, 3, 4, 5]),
        School(id=1, capacity=1, preferences=[0, 1, 2, 3, 4, 5]),
        School(id=2, capacity=1, preferences=[0, 1, 2, 3, 4, 5]),
        School(id=3, capacity=1, preferences=[0, 1, 2, 3, 4, 5]),
        School(id=4, capacity=1, preferences=[0, 1, 2, 3, 4, 5]),
        School(id=5, capacity=1, preferences=[0, 1, 2, 3, 4, 5]),
    ]

    matching = deferred_acceptance(students, schools)
    assert matching.matches == {
        students[0]: schools[5],
        students[1]: schools[4],
        students[2]: schools[3],
        students[3]: schools[2],
        students[4]: schools[1],
        students[5]: schools[0],
    }
    assert find_unstable_pair(students, schools, matching) is None


def test_deferred_acceptance_capacity_2():
    students = [
        Student(id=0, preferences=[5, 4, 3, 2, 1, 0]),
        Student(id=1, preferences=[5, 4, 3, 2, 1, 0]),
        Student(id=2, preferences=[5, 4, 3, 2, 1, 0]),
        Student(id=3, preferences=[5, 4, 3, 2, 1, 0]),
        Student(id=4, preferences=[5, 4, 3, 2, 1, 0]),
        Student(id=5, preferences=[5, 4, 3, 2, 1, 0]),
    ]

    schools = [
        School(id=0, capacity=2, preferences=[0, 1, 2, 3, 4, 5]),
        School(id=1, capacity=2, preferences=[0, 1, 2, 3, 4, 5]),
        School(id=2, capacity=2, preferences=[0, 1, 2, 3, 4, 5]),
        School(id=3, capacity=2, preferences=[0, 1, 2, 3, 4, 5]),
        School(id=4, capacity=2, preferences=[0, 1, 2, 3, 4, 5]),
        School(id=5, capacity=2, preferences=[0, 1, 2, 3, 4, 5]),
    ]

    matching = deferred_acceptance(students, schools)
    assert matching.matches == {
        students[0]: schools[5],
        students[1]: schools[5],
        students[2]: schools[4],
        students[3]: schools[4],
        students[4]: schools[3],
        students[5]: schools[3],
    }
    assert find_unstable_pair(students, schools, matching) is None


def test_deferred_acceptance_limited_student_preference_lists():
    students = [
        Student(id=0, preferences=[5, 4]),
        Student(id=1, preferences=[5, 4]),
        Student(id=2, preferences=[5, 4]),
        Student(id=3, preferences=[5, 4]),
        Student(id=4, preferences=[5, 4]),
        Student(id=5, preferences=[5, 4]),
    ]

    schools = [
        School(id=0, capacity=2, preferences=[0, 1, 2, 3, 4, 5]),
        School(id=1, capacity=2, preferences=[0, 1, 2, 3, 4, 5]),
        School(id=2, capacity=2, preferences=[0, 1, 2, 3, 4, 5]),
        School(id=3, capacity=2, preferences=[0, 1, 2, 3, 4, 5]),
        School(id=4, capacity=2, preferences=[0, 1, 2, 3, 4, 5]),
        School(id=5, capacity=2, preferences=[0, 1, 2, 3, 4, 5]),
    ]

    matching = deferred_acceptance(students, schools)
    assert matching.matches == {
        students[0]: schools[5],
        students[1]: schools[5],
        students[2]: schools[4],
        students[3]: schools[4],
    }
    assert matching.unassigned == {
        students[4],
        students[5],
    }
    assert find_unstable_pair(students, schools, matching) is None
