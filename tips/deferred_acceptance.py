"""An implementation of the student-proposing deferred acceptance algorithm."""
from dataclasses import dataclass
from dataclasses import field
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple
import heapq
import itertools


@dataclass
class Student:
    id: int
    """Preferences of School.id, from highest priority to lowest priority."""
    preferences: List[int]
    """The highest priority school this student has yet to be rejected from."""
    best_unrejected: int = 0

    def __hash__(self):
        return self.id


@dataclass
class School:
    id: int
    """Preferences of Student.id, from highest priority to lowest priority."""
    preferences: List[int]
    """The number of open seats at the school."""
    capacity: int
    """A list of size at most self.capacity containing held applications."""
    held: List[int] = field(default_factory=list)


@dataclass
class Matching:
    matches: Dict[Student, School]
    unassigned: Set[Student] = field(default_factory=set)


def run_round(
    students: Dict[int, Student],
    schools: Dict[int, School],
    to_apply: Iterable[Student],
) -> Set[Student]:
    """Run one round of deferred acceptance, returning a list of rejections."""
    applications: Dict[int, List[int]] = {
        id: school.held[:] for (id, school) in schools.items()
    }

    for student in to_apply:
        chosen_school = schools[student.preferences[student.best_unrejected]]
        applications[chosen_school.id].append(student.id)

    new_held_students = {
        id: heapq.nsmallest(
            schools[id].capacity, app_ids, key=schools[id].preferences.index,
        )
        for (id, app_ids) in applications.items()
    }

    rejections = set()
    for (id, school) in schools.items():
        school.held = new_held_students[id]
        rejections |= set(applications[id]) - set(new_held_students[id])

    return set(students[student_id] for student_id in rejections)


def deferred_acceptance(
    students: Iterable[Student], schools: Iterable[School],
) -> Matching:
    """Construct a stable matching of students to schools.

    Arguments:
        students: the Students to match
        schools: the Schools to match

    Returns:
        A dict {Student: School} assigning each student to a school.
    """
    to_apply = set(students)
    unassigned: Set[Student] = set()
    student_index = {student.id: student for student in students}
    school_index = {school.id: school for school in schools}

    while to_apply:
        rejections = run_round(student_index, school_index, to_apply)
        for student in rejections:
            student.best_unrejected += 1

        unassigned = unassigned.union(
            {
                student
                for student in rejections
                if student.best_unrejected >= len(student.preferences)
            },
        )
        to_apply = rejections - unassigned

    return Matching(
        matches={
            student_index[student_id]: school
            for school in schools
            for student_id in school.held
        },
        unassigned=unassigned,
    )


def find_unstable_pair(
    students: Iterable[Student], schools: Iterable[School], matching: Matching,
) -> Optional[Tuple[Student, School]]:
    """Returns an unstable pair in the matching, or None if none exists."""

    def precedes(L, item1, item2):
        return L.index(item1) < L.index(item2)

    def student_prefers(student, school):
        # Returns true if a student prefers the input school over their
        # assigned school
        try:
            return precedes(
                student.preferences, school.id, matching.matches[student].id,
            )
        except ValueError:
            # If the student doesn't have the school in their list, treat as
            # a non-preference.
            return False

    def school_prefers(school, student):
        # Returns true if a school prefers the input student over at least one
        # of their assigned students.
        assigned_students = [
            student for (student, sch) in matching.matches.items() if sch == school
        ]
        student_cmps = [
            precedes(school.preferences, student.id, assigned_student.id)
            for assigned_student in assigned_students
        ]
        return any(student_cmps)

    for (student, school) in itertools.product(students, schools):
        if student not in matching.unassigned:
            if (
                school != matching.matches[student]
                and student_prefers(student, school)
                and school_prefers(school, student)
            ):
                return (student, school)

    return None
