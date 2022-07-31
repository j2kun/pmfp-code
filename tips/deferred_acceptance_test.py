import hypothesis
from hypothesis import given
from hypothesis.strategies import composite
from hypothesis.strategies import integers
from hypothesis.strategies import permutations

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
@hypothesis.settings(print_blob=True)
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
